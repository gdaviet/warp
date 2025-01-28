# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch
from embedded_sim_utils import embed_tri_mesh, flexicubes_from_sdf_grid, sim_from_flexicubes, surface_positions
from kaolin.io import import_mesh
from softbody_sim import ClassicFEM

import warp as wp
import warp.fem as fem


def load_mesh(path):
    """Load and normalize an obj mesh from path"""

    mesh = import_mesh(path, triangulate=True).cuda()
    # normalize to [-1, 1]
    half_bbox = (
        0.5 * torch.min(mesh.vertices, dim=0)[0],
        0.5 * torch.max(mesh.vertices, dim=0)[0],
    )
    normalized_vertices = (mesh.vertices - half_bbox[0] - half_bbox[1]) / torch.max(half_bbox[1] - half_bbox[0] + 0.001)
    return wp.Mesh(
        wp.from_torch(normalized_vertices, dtype=wp.vec3),
        wp.from_torch(mesh.faces.flatten().to(torch.int32)),
        support_winding_number=True,
    )


@fem.integrand
def fixed_points_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u_cur: fem.Field,
    u: fem.Field,
    v: fem.Field,
):
    """Dirichlet boundary condition projector

    Here we simply clamp points near Z boundaries
    """

    y = domain(s)

    clamped = wp.select(wp.abs(y[1]) > 0.9, 0.0, 1.0)

    return wp.dot(u(s), v(s)) * clamped


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
    """Utility struct for storing simulation state"""

    def __init__(self, geo):
        self.geo = geo

        self.sim = None
        self.tri_mesh = None
        self.tri_vtx_quadrature = None
        self.rest_points = None

    def create_sim(self, flexicubes):
        if self.sim is not None:
            # save previous displacement and velocity
            prev_displacement_field = self.sim.u_field.space.make_field()
            prev_velocity_field = self.sim.du_field.space.make_field()
            fem.interpolate(self.sim.u_field, dest=prev_displacement_field)
            fem.interpolate(self.sim.du_field, dest=prev_velocity_field)
            prev_displacement = prev_displacement_field.dof_values
            prev_velocity = prev_velocity_field.dof_values
        else:
            prev_displacement = None
            prev_velocity = None

        # (Re)create simulation
        self.sim = sim_from_flexicubes(ClassicFEM, flexicubes, geo, args)
        self.sim.set_fixed_points_condition(
            fixed_points_projector_form,
        )
        self.sim.init_constant_forms()
        self.sim.project_constant_forms()

        # Interpolate back previous displacement
        if prev_displacement is not None:
            prev_displacement_field = self.sim.u_field.space.make_field()
            prev_displacement_field.dof_values = prev_displacement
            prev_velocity_field = self.sim.du_field.space.make_field()
            prev_velocity_field.dof_values = prev_velocity
            fem.interpolate(prev_displacement_field, dest=self.sim.u_field)
            fem.interpolate(prev_velocity_field, dest=self.sim.du_field)

        # Embed triangle mesh
        fc_bd_cubes = flexicubes["fc_bd_cubes"]
        fc_bd_nv = flexicubes["fc_bd_nv"]
        tri_vertices = flexicubes["tri_vertices"]
        tri_faces = wp.array(flexicubes["tri_faces"], dtype=int).flatten()
        tri_vtx_pos = wp.array(tri_vertices, dtype=wp.vec3)

        self.tri_mesh = wp.Mesh(tri_vtx_pos, tri_faces)
        self.rest_points = wp.clone(tri_vtx_pos)
        self.surface_vtx_quadrature = embed_tri_mesh(self.sim.u_test.domain, tri_vtx_pos, fc_bd_cubes, fc_bd_nv)

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


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})
    wp.set_module_options({"fast_math": True})

    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("--quadrature_model", "-qm", default=None)
    parser.add_argument("--force_scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=64)

    ClassicFEM.add_parser_arguments(parser)
    args = parser.parse_args()

    # fall back to full-cell quadrature if neural model not provided
    args.reg_qp = 2 if args.quadrature_model is None else 0
    args.clip = False

    res = args.resolution
    geo = fem.Grid3D(res=wp.vec3i(res), bounds_lo=wp.vec3(-1), bounds_hi=wp.vec3(1))

    # sample mesh SDF on grid nodes
    source_mesh = load_mesh(args.mesh)
    grid_node_pos = fem.make_polynomial_space(geo).node_positions()
    grid_sdf = wp.empty(grid_node_pos.shape[0], dtype=float)
    wp.launch(
        mesh_sdf_kernel,
        dim=grid_node_pos.shape,
        inputs=[source_mesh.id, grid_node_pos, grid_sdf],
    )
    grid_sdf = wp.to_torch(grid_sdf)
    grid_node_pos = wp.to_torch(grid_node_pos)

    # Create
    flexicubes = flexicubes_from_sdf_grid(res, pos=grid_node_pos, sdf=grid_sdf, sdf_grad_func=None)

    # Create simulation
    clay = Clay(geo)
    clay.create_sim(flexicubes)

    # Add hooks for displaying surface and run sim

    def init_surface(flexicubes):
        tri_vertices = flexicubes["tri_vertices"]
        tri_faces = flexicubes["tri_faces"]

        surface = ps.register_surface_mesh("surf", tri_vertices, tri_faces)
        surface.set_edge_width(1.0)

    prev_world_pos = None

    # user interface callback
    def callback():
        global grid_sdf, prev_world_pos, force_center_quadrature

        io = psim.GetIO()

        if io.KeyMods in (1, psim.ImGuiKeyModFlags_Ctrl):
            # ctrl + mouse: update SDF values

            if io.MouseDown[0]:
                sign = -1.0  # left-mouse, add material
            elif io.MouseDown[1]:
                sign = 1.0  # right-mouse, remove material
            else:
                sign = 0.0

            if sign != 0.0:
                screen_coords = io.MousePos

                # Convert clicked position to rest pose
                world_pos = ps.screen_coords_to_world_position(screen_coords)
                if np.all(np.isfinite(world_pos)):
                    prev_world_pos = world_pos
                elif prev_world_pos is not None:
                    world_pos = prev_world_pos

                rest_pos = wp.to_torch(clay.world_to_rest_pos(world_pos))

                # locally update sdf values
                delta_pos_sq = torch.sum((grid_node_pos - rest_pos) * (grid_node_pos - rest_pos), dim=1)
                delta_sdf = torch.exp(-0.25 * delta_pos_sq * res * res)

                grid_sdf += 50.0 * sign / res * delta_sdf

                # rebuilds flexicubes structure and recreate sim
                flexicubes = flexicubes_from_sdf_grid(res, grid_sdf, grid_node_pos)
                clay.create_sim(flexicubes)
                init_surface(flexicubes)

                io.WantCaptureMouse = True

        sim = clay.sim

        # run one frame of simulation
        sim.run_frame()

        # Interpolate deformation back to vertices
        fem.interpolate(
            surface_positions,
            quadrature=clay.surface_vtx_quadrature,
            dest=clay.tri_mesh.points,
            fields={"displacement": sim.u_field},
        )
        surf_mesh = ps.get_surface_mesh("surf")
        surf_mesh.update_vertex_positions(clay.tri_mesh.points.numpy())

        # Dynamic picking force
        # (shift + click)

        if io.KeyMods in (2, psim.ImGuiKeyModFlags_Shift):  # shift
            if io.MouseClicked[0]:
                screen_coords = io.MousePos
                world_ray = ps.screen_coords_to_world_ray(screen_coords)
                world_pos = ps.screen_coords_to_world_position(screen_coords)

                if np.all(np.isfinite(world_pos)):
                    rest_pos = clay.world_to_rest_pos(world_pos)

                    # update force application point
                    sim.forces.count = 1
                    sim.forces.forces.zero_()
                    sim.forces.centers = rest_pos
                    sim.update_force_weight()

                    # embed force center so we can move it with the sim
                    force_center_quadrature = fem.PicQuadrature(
                        fem.Cells(geo),
                        rest_pos,
                    )
                    force_center_quadrature._domain = sim.u_test.domain

                else:
                    sim.forces.count = 0

            elif sim.forces.count > 0:
                screen_coords = io.MousePos
                world_ray = ps.screen_coords_to_world_ray(screen_coords)

                # interpolate current position of force application center
                force_center_position = wp.empty(shape=(1,), dtype=wp.vec3)
                fem.interpolate(
                    surface_positions,
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

                sim.forces.forces = wp.array([perp * args.force_scale], dtype=wp.vec3)

                # force line visualization
                ps.get_curve_network("force_line").update_node_positions(
                    np.array([deformed_force_center, deformed_force_center + perp])
                )
                ps.get_curve_network("force_line").set_enabled(True)

            if io.MouseReleased[0]:
                sim.forces.count = 0
                ps.get_curve_network("force_line").set_enabled(False)

            io.WantCaptureMouse = sim.forces.count > 0

    ps.init()

    ps.set_ground_plane_mode(mode_str="none")
    ps.register_curve_network(
        "force_line",
        nodes=np.zeros((2, 3)),
        edges=np.array([[0, 1]]),
        enabled=False,
    )

    init_surface(flexicubes)

    # ps.set_build_default_gui_panels(False)
    ps.set_user_callback(callback)
    ps.show()
