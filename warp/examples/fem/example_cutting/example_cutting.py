#! /usr/bin/env uv run --script

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kaolin==0.17.0",
#     "polyscope==2.1",
#     "torch==2.5.1",
#     "warp-lang>=1.9.0dev20250801",
#     "torchvision",
#     "trimesh",
#     "meshio",
# ]
# [tool.uv]
# find-links = ["https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html"]
# [tool.uv.sources]
# warp-lang = { index = "nvidia"}
# [[tool.uv.index]]
# name = "nvidia"
# url = "https://pypi.nvidia.com"
# explicit = true
# ///

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse

import numpy as np
from mfem.collisions import CollisionPotential, MeshSelfCollisionHandler
from mfem.mfem_3d import MFEM_RS_F, MFEM_sF_S
from mfem.potentials import VolumetricForcePotential
from mfem.softbody_sim import ClassicFEM
from utils.embedded_sim_utils import (
    flexicubes_from_sdf_grid,
    sim_from_flexicubes,
)

import warp as wp
import warp.fem as fem


def load_normalized_mesh(path):
    """Load and normalize an obj mesh from path"""

    from warp.sim.utils import load_mesh

    points, indices = load_mesh(path)
    bbox_min, bbox_max = (
        np.min(points, axis=0),
        np.max(points, axis=0),
    )
    normalized_vertices = (2.0 * points - bbox_min - bbox_max) / np.max(bbox_max - bbox_min + 0.001)
    return wp.Mesh(
        wp.array(normalized_vertices, dtype=wp.vec3),
        wp.array(indices, dtype=int),
        support_winding_number=True,
    )


@fem.integrand
def fixed_points_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u_cur: fem.Field,
    u: fem.Field,
    v: fem.Field,
    y_min: float,
    y_max: float,
):
    """Dirichlet boundary condition projector

    Here we simply clamp points near Y boundaries
    """

    y = domain(s)[1]
    clamped = wp.where(y > y_max or y < y_min, 1.0, 0.0)

    return wp.dot(u(s), v(s)) * clamped


@fem.integrand
def deformed_position(s: fem.Sample, domain: fem.Domain, displacement: fem.Field):
    return domain(s) + displacement(s)


@wp.kernel
def world_to_rest_pose_kernel(
    mesh: wp.uint64,
    rest_points: wp.array(dtype=wp.vec3),
    pos: wp.vec3,
    out: wp.array(dtype=wp.vec3),
):
    """
    Converts a point on the deformed surface to its rest-pose counterpart
    """

    max_dist = 1.0
    query = wp.mesh_query_point_no_sign(mesh, pos, max_dist)

    if query.result:
        faces = wp.mesh_get(mesh).indices
        v0 = rest_points[faces[3 * query.face + 0]]
        v1 = rest_points[faces[3 * query.face + 1]]
        v2 = rest_points[faces[3 * query.face + 2]]

        p = v0 + query.u * (v1 - v0) + query.v * (v2 - v0)

    else:
        p = pos

    out[0] = p


@wp.kernel
def sculpt_sdf(
    amount: float,
    falloff: float,
    update_pos: wp.array(dtype=wp.vec3),
    grid_node_pos: wp.array(dtype=wp.vec3),
    grid_sdf: wp.array(dtype=float),
):
    """Update grid sdf values around a given position"""

    i = wp.tid()
    dist_sq = wp.length_sq(grid_node_pos[i] - update_pos[0])

    delta_sdf = wp.exp(-falloff * dist_sq)
    grid_sdf[i] += amount * delta_sdf


@wp.kernel
def mesh_sdf_kernel(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    sdf: wp.array(dtype=float),
):
    """Builds an SDF using mesh closest-point queries"""

    i = wp.tid()
    pos = points[i]

    max_dist = 1.0
    query = wp.mesh_query_point_sign_winding_number(mesh, pos, max_dist)

    if query.result:
        mesh_pos = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        sdf[i] = query.sign * wp.length(pos - mesh_pos)
    else:
        sdf[i] = 1.0


class Clay:
    """Utility struct for storing simulation state and handling dynamic simulation topology"""

    def __init__(self, geo):
        self.geo = geo

        self.sim = None
        self.tri_mesh = None
        self.tri_vtx_quadrature = None
        self.rest_points = None

        self._sim_initialized = False

    def create_sim(self, flexicubes, sim_class):
        prev_sim = self.sim

        # (Re)create simulation
        self.sim = sim_from_flexicubes(
            sim_class,
            flexicubes,
            geo,
            args,
            quadrature_model=args.quadrature_model,
        )

        self._sim_initialized = False

        # Interpolate back previous displacement
        if prev_sim is not None:
            new_domain = self.sim.u_test.domain
            prev_displacement_field = fem.NonconformingField(new_domain, prev_sim.u_field)
            prev_velocity_field = fem.NonconformingField(new_domain, prev_sim.du_field)
            fem.interpolate(
                prev_displacement_field,
                dest=self.sim.u_field,
                kernel_options={"enable_backward": False},
            )
            fem.interpolate(
                prev_velocity_field,
                dest=self.sim.du_field,
                kernel_options={"enable_backward": False},
            )

        # Embed triangle mesh
        tri_vertices = flexicubes.tri_vertices
        tri_faces = wp.array(flexicubes.tri_faces, dtype=int).flatten()
        tri_vtx_pos = wp.array(tri_vertices, dtype=wp.vec3)

        self.tri_mesh = wp.Mesh(tri_vtx_pos, tri_faces)
        self.rest_points = wp.clone(tri_vtx_pos)
        self.surface_vtx_quadrature = fem.PicQuadrature(self.sim.u_test.domain, tri_vtx_pos, max_dist=4.0 / res)

        self.collision_handler = MeshSelfCollisionHandler(self.surface_vtx_quadrature, self.tri_mesh)
        if not self.sim.args.matrix_free:
            # Matrix free sim does not handle collisions yet
            collision_potential = CollisionPotential(self.sim, self.collision_handler)
            self.sim.add_energy_potential(collision_potential)

        self.volumetric_forces = VolumetricForcePotential(self.sim, reserve_count=1)
        self.volumetric_forces.forces.radii.fill_(2.0 / res)
        self.sim.add_energy_potential(self.volumetric_forces)

    def is_initialized(self):
        return self._sim_initialized

    def ensure_sim_is_initialized(self):
        if not self._sim_initialized:
            self.sim.set_fixed_points_condition(
                fixed_points_projector_form,
                {
                    "y_min": self.sim.args.y_min,
                    "y_max": self.sim.args.y_max,
                },
            )

            self.sim.init_constant_forms()
            self.sim.project_constant_forms()
            self._sim_initialized = True

    def world_to_rest_pos(self, world_pos):
        tri_mesh = self.tri_mesh

        rest_pos = wp.empty(1, dtype=wp.vec3)
        tri_mesh.refit()
        wp.launch(
            world_to_rest_pose_kernel,
            dim=1,
            inputs=[tri_mesh.id, self.rest_points, world_pos, rest_pos],
        )
        return rest_pos


def setup_interactive_viewer(clay: Clay, grid_node_pos: wp.array, grid_sdf: wp.array):
    """Setups an interactive polyscope viewer and register hooks for sculpting and picking"""

    import polyscope as ps
    import polyscope.imgui as psim

    # Add hooks for displaying surface and run sim

    def register_ps_meshes(flexicubes, sim, first_frame=False):
        tri_vertices = flexicubes.tri_vertices
        tri_faces = flexicubes.tri_faces

        surface = ps.register_surface_mesh("surf", tri_vertices, tri_faces)
        surface.set_edge_width(1.0)

    prev_world_pos = None
    frame_id = 0
    force_center_quadrature = None

    # user interface callback
    def callback():
        nonlocal prev_world_pos, force_center_quadrature, frame_id

        io = psim.GetIO()

        ctrl = getattr(psim, "ImGuiKeyModFlags_Ctrl", None) or psim.ImGuiModFlags_Ctrl
        shift = getattr(psim, "ImGuiKeyModFlags_Shift", None) or psim.ImGuiModFlags_Shift

        sculpting = False
        if io.KeyMods in (1, ctrl):
            # ctrl + mouse: update SDF values

            if io.MouseDown[0]:
                sign = -1.0  # left-mouse, add material
                sculpting = True
            elif io.MouseDown[1]:
                sign = 1.0  # right-mouse, remove material
                sculpting = True

            if sculpting:
                screen_coords = io.MousePos

                # Convert clicked position to rest pose
                world_pos = ps.screen_coords_to_world_position(screen_coords)
                if np.all(np.isfinite(world_pos)):
                    prev_world_pos = world_pos
                elif prev_world_pos is not None:
                    world_pos = prev_world_pos

                rest_pos = clay.world_to_rest_pos(world_pos)

                amount = 50.0 * sign / res
                falloff = 0.25 * res * res

                wp.launch(
                    sculpt_sdf,
                    dim=grid_node_pos.shape[0],
                    inputs=[amount, falloff, rest_pos, grid_node_pos, grid_sdf],
                )

                # rebuilds flexicubes structure and recreate sim
                fc_data = flexicubes_from_sdf_grid(res, grid_sdf, grid_node_pos)
                clay.create_sim(fc_data, sim_class=sim_class)
                register_ps_meshes(fc_data, clay.sim)

                io.WantCaptureMouse = True

        sim = clay.sim

        # run one frame of simulation
        if not sculpting:
            if args.n_frames >= 0 and frame_id >= args.n_frames:
                return
            frame_id += 1

            clay.ensure_sim_is_initialized()
            with wp.ScopedTimer(f"Frame {frame_id}", synchronize=True):
                sim.run_frame()

        # Interpolate deformation back to vertices
        fem.interpolate(
            deformed_position,
            quadrature=clay.surface_vtx_quadrature,
            dest=clay.tri_mesh.points,
            fields={"displacement": sim.u_field},
        )
        surf_mesh = ps.get_surface_mesh("surf")
        surf_mesh.update_vertex_positions(clay.tri_mesh.points.numpy())

        if clay.collision_handler and clay.is_initialized():
            ps.register_point_cloud(
                "CP",
                clay.collision_handler.cp_world_position().numpy()[clay.tri_mesh.points.shape[0] :],
            )

        # Dynamic picking force
        # (shift + click)

        if io.KeyMods in (2, shift):  # shift
            if io.MouseClicked[0]:
                screen_coords = io.MousePos
                world_ray = ps.screen_coords_to_world_ray(screen_coords)
                world_pos = ps.screen_coords_to_world_position(screen_coords)

                if np.all(np.isfinite(world_pos)):
                    rest_pos = clay.world_to_rest_pos(world_pos)

                    # update force application point
                    clay.volumetric_forces.forces.count = 1
                    clay.volumetric_forces.forces.forces.zero_()
                    clay.volumetric_forces.forces.centers = rest_pos
                    clay.volumetric_forces.update_force_weight()

                    # embed force center so we can move it with the sim
                    force_center_quadrature = fem.PicQuadrature(fem.Cells(sim.geo), rest_pos, max_dist=2.0 / res)
                    force_center_quadrature._domain = sim.u_test.domain

                else:
                    clay.volumetric_forces.forces.count = 0

            elif clay.volumetric_forces.forces.count > 0:
                screen_coords = io.MousePos
                world_ray = ps.screen_coords_to_world_ray(screen_coords)

                # interpolate current position of force application center
                force_center_position = wp.empty(shape=(1,), dtype=wp.vec3)
                fem.interpolate(
                    deformed_position,
                    quadrature=force_center_quadrature,
                    dest=force_center_position,
                    fields={"displacement": sim.u_field},
                )
                deformed_force_center = force_center_position.numpy()[0]

                # update picking force direction
                ray_dir = world_ray / np.linalg.norm(world_ray)
                ray_orig = ps.get_view_camera_parameters().get_position()
                perp = ray_orig - deformed_force_center
                perp -= np.dot(perp, ray_dir) * ray_dir

                clay.volumetric_forces.forces.forces = wp.array([perp * args.force_scale], dtype=wp.vec3)

                # force line visualization
                ps.get_curve_network("force_line").update_node_positions(
                    np.array([deformed_force_center, deformed_force_center + perp])
                )
                ps.get_curve_network("force_line").set_enabled(True)

            if io.MouseReleased[0]:
                clay.volumetric_forces.forces.count = 0
                ps.get_curve_network("force_line").set_enabled(False)

            io.WantCaptureMouse = clay.volumetric_forces.forces.count > 0

    ps.init()

    ps.set_ground_plane_mode(mode_str="none")
    ps.register_curve_network(
        "force_line",
        nodes=np.zeros((2, 3)),
        edges=np.array([[0, 1]]),
        enabled=False,
    )

    register_ps_meshes(fc_data, clay.sim, first_frame=True)

    # ps.set_build_default_gui_panels(False)
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": False})
    wp.set_module_options({"fast_math": True})

    class_parser = argparse.ArgumentParser()
    class_parser.add_argument(
        "--variant",
        "-v",
        choices=["mfem", "classic", "trusty22"],
        default="classic",
    )
    class_args, remaining_args = class_parser.parse_known_args()

    if class_args.variant == "mfem":
        sim_class = MFEM_RS_F
    elif class_args.variant == "trusty22":
        sim_class = MFEM_sF_S
    else:
        sim_class = ClassicFEM

    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument(
        "--quadrature_model",
        "-qm",
        default=None,
        nargs="*",
        help="Path to the saved neural quadrature MLP weights. If not provided, use regular quadrature",
    )
    parser.add_argument("--resolution", type=int, default=64, help="Grid resolution (at finest level)")
    parser.add_argument(
        "--force_scale",
        type=float,
        default=1.0,
        help="Scaling factor for dynamic picking forces",
    )
    parser.add_argument(
        "--y_min",
        type=float,
        default=-0.9,
        help="Clamp points below this Y value",
    )
    parser.add_argument(
        "--y_max",
        type=float,
        default=0.9,
        help="Clamp points above this Y value",
    )

    sim_class.add_parser_arguments(parser)
    MeshSelfCollisionHandler.add_parser_arguments(parser)
    args = parser.parse_args(remaining_args)

    # fall back to full-cell quadrature if neural model not provided
    args.clip = False
    args.ground_height = -1
    args.collision_radius = 0.5 / args.resolution

    res = args.resolution

    # Regular grid for evaluating sdf
    geo = fem.Grid3D(res=wp.vec3i(res), bounds_lo=wp.vec3(-1), bounds_hi=wp.vec3(1))

    # sample mesh SDF on grid nodes
    source_mesh = load_normalized_mesh(args.mesh)
    grid_node_pos = fem.make_polynomial_space(geo).node_positions()
    grid_sdf = wp.empty(grid_node_pos.shape[0], dtype=float)
    wp.launch(
        mesh_sdf_kernel,
        dim=grid_node_pos.shape,
        inputs=[source_mesh.id, grid_node_pos, grid_sdf],
    )

    # Create flexicube data from sdf grid
    fc_data = flexicubes_from_sdf_grid(res, grid_node_pos=grid_node_pos, grid_node_sdf=grid_sdf, sdf_grad_func=None)

    # Create simulation
    clay = Clay(geo)
    clay.create_sim(fc_data, sim_class=sim_class)

    # Setup interactive viewer and run simulation
    setup_interactive_viewer(clay, grid_node_pos, grid_sdf)
