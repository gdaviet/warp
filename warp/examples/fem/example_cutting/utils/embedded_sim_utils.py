# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utilities for setting-up embedded simulation of surfaces in hex meshes"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

import warp as wp
import warp.fem as fem
from warp.examples.fem.utils import gen_hexmesh

from .voxel_quadratures import neural_quadrature, regular_quadrature

__all__ = [
    "FcData",
    "flexicubes_from_sdf_grid",
    "get_quadrature",
    "sim_from_flexicubes",
]

FC_WEIGHT_SCALE = 0.95
"""alpha-parameter scaling for Flexicubes"""


@dataclass
class FcData:
    """Utility struct for storing a FlexiCubes grid definition and extracted geometry"""

    # Implicit geometry
    cubes: np.ndarray
    """Vertex indices for each grid cell, z-major"""
    pos: np.ndarray
    """per-grid-node sdf value"""
    sdf: np.ndarray
    """position of grid nodes"""
    weights: Optional[np.ndarray] = None
    """(alpha, beta, gamme) FlexiCubes parameters, may be None"""
    stiffness: Optional[np.ndarray] = None
    """Per-vertex stiffness scaling. May be None"""

    # Extracted geometry
    tet_vertices: Optional[np.ndarray] = None
    """Tetrahedron vertices. May be None"""
    tet_indices: Optional[np.ndarray] = None
    """Tetrahedron indices. May be None"""
    tri_vertices: Optional[np.ndarray] = None
    """Triangle vertices. May be None"""
    tri_faces: Optional[np.ndarray] = None
    """Triangle faces. May be None"""

    # Simulation state
    vtx_displ: Optional[np.ndarray] = None
    """Vertex displacements. May be None"""


def get_quadrature(model_path, cell_vtx, grid_sdf, cell_weights, clip=True, order=0, device="cuda"):
    if cell_weights is None:
        cell_alphas = np.ones(cell_vtx.shape, dtype=float)
    else:
        cell_alphas = 1.0 + FC_WEIGHT_SCALE * np.tanh(cell_weights[:, 12:20])

    if model_path is None:
        return regular_quadrature(cell_vtx, grid_sdf, cell_alphas, clip=clip, order=order)

    return neural_quadrature(model_path, cell_vtx, grid_sdf, cell_alphas)


def sim_from_flexicubes(
    sim_class,
    fc_data: FcData,
    geo: fem.Geometry,
    sim_args,
    quadrature_model: Optional[Union[str, List[str]]] = None,
    clip_qp: bool = False,
    quadrature_order: int = 0,
):
    """Instantiates a simulator instance from serialized Flexicubes data"""

    # Compute quadrature and active cells from flexicube sdf
    if quadrature_model:
        if isinstance(quadrature_model, str):
            quad_model = quadrature_model
        else:
            quad_model = quadrature_model[0]
    else:
        quad_model = None

    quad_order = 2 * sim_args.degree if quadrature_order == 0 else quadrature_order

    qc, qw, active_cells = get_quadrature(
        quad_model,
        fc_data.cubes,
        fc_data.sdf,
        fc_data.weights,
        clip=clip_qp,
        order=quad_order,
    )

    grid_displacement_field = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3).make_field()
    grid_displacement_field.dof_values = fc_data.pos
    grid_displacement_field.dof_values.requires_grad = True

    deformed_grid = grid_displacement_field.make_deformed_geometry(relative=False)
    deformed_grid.build_bvh()

    # Initialize sim
    sim = sim_class(deformed_grid, active_cells, sim_args)
    sim.init_displacement_space()

    if fc_data.stiffness is not None:
        sim.scale_lame_field(wp.array(fc_data.stiffness, dtype=float))

    # Replace regular quadrature will learned quadrature
    domain = fem.Cells(sim.geo_partition)
    if sim.cells is not None:
        domain_qc = qc[sim.cells.array].contiguous()
        domain_qw = qw[sim.cells.array].contiguous()
        quadrature = fem.ExplicitQuadrature(domain, domain_qc, domain_qw)
    else:
        quadrature = fem.ExplicitQuadrature(domain, qc, qw)

    sim.vel_quadrature = quadrature
    sim.strain_quadrature = quadrature
    sim.elasticity_quadrature = quadrature

    # For Mixed FEM: locate strain nodes at quadrature points
    geo_quadrature = fem.ExplicitQuadrature(fem.Cells(deformed_grid), qc, qw)
    # basis evaluated only at quadrature points, PointBasisSpace is ok
    rbf_basis = fem.PointBasisSpace(geo_quadrature)
    sim.set_strain_basis(rbf_basis)

    sim.init_strain_spaces()

    return sim


def flexicubes_from_sdf_grid(
    res,
    grid_node_sdf,
    grid_node_pos,
    sdf_grad_func=None,
    output_tetmesh=False,
    device="cuda",
):
    """Creates and serialize a Flexicubes datastructure from a SDF discretized on a dense grid"""

    try:
        import torch
        from kaolin.non_commercial import FlexiCubes

        grid_node_sdf = wp.to_torch(grid_node_sdf)
        grid_node_pos = wp.to_torch(grid_node_pos)

        fc = FlexiCubes(device)
        _x_nx3, cube_fx8 = fc.construct_voxel_grid(res)

        weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device=device)

        flexi = FcData(
            pos=grid_node_pos.detach().cpu().numpy(),
            sdf=grid_node_sdf.detach().cpu().numpy(),
            cubes=cube_fx8.detach().cpu().numpy(),
            weights=weight.detach().cpu().numpy(),
            stiffness=None,
        )

        vertices, faces, L_dev = fc(
            voxelgrid_vertices=grid_node_pos,
            scalar_field=grid_node_sdf,
            cube_idx=cube_fx8,
            resolution=res,
            weight_scale=FC_WEIGHT_SCALE,
            beta=weight[:, :12],
            alpha=weight[:, 12:20],
            gamma_f=weight[:, 20],
            training=False,
            output_tetmesh=output_tetmesh,
            grad_func=sdf_grad_func,
        )

        if output_tetmesh:
            flexi.tet_vertices = vertices.detach().cpu().numpy()
            flexi.tet_indices = faces.detach().cpu().numpy()
            flexi.vtx_displ = np.zeros(vertices.shape, dtype=np.float32)
        else:
            flexi.tri_vertices = vertices.detach().cpu().numpy()
            flexi.tri_faces = faces.detach().cpu().numpy()
            flexi.vtx_displ = np.zeros(vertices.shape, dtype=np.float32)

        return flexi

    except ImportError:
        wp.utils.warn("Failed to import kaolin flexicubes falling back wp wp.MarchingCubes")

        assert not output_tetmesh

        mc = wp.MarchingCubes(res + 1, res + 1, res + 1, res**3, res**3)
        mc.surface(grid_node_sdf.reshape((res + 1, res + 1, res + 1)), threshold=0.0)
        _vtx, hexes = gen_hexmesh(wp.vec3i(res))
        hexes = hexes.numpy()[:, [0, 4, 2, 6, 1, 5, 3, 7]]  # make z-major for consistency with flexicubes

        flexi = FcData(
            pos=grid_node_pos.numpy(),
            sdf=grid_node_sdf.numpy(),
            cubes=hexes,
            weights=None,
            stiffness=None,
            tri_vertices=2.0 * mc.verts.numpy() / (res + 1) - 1.0,
            tri_faces=mc.indices.numpy().reshape(-1, 3),
            vtx_displ=None,
        )
        return flexi
