# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Utilities for setting-up embedded simulation of surfaces in hex meshes"""

import functools

import numpy as np
import torch
from kaolin.non_commercial import FlexiCubes

import warp as wp
import warp.fem as fem

FC_WEIGHT_SCALE = 0.95
"""alpha-parameter scaling for Flexicubes"""


def infer_quadrature(model, cube, sdf, weight):
    """Inferred quadrature points from MLP"""

    cell_sdf = sdf[cube]
    cell_alpha = 1.0 + FC_WEIGHT_SCALE * torch.tanh(weight[:, 12:20])

    qc, qw = model(cell_sdf * cell_alpha)

    qc = qc.float().flip(dims=(2,))  # flip because FC cube corners are z-major

    min_sdf, _ = torch.min(cell_sdf, dim=1)
    active_cells = torch.where(min_sdf < 0, 1, 0)

    qc = qc.contiguous()
    qw = (qw + 1.0e-8).contiguous()

    return qc, qw, active_cells


def regular_quadrature(cube, sdf, weight, clip=True, order=2):
    """Regular Gauss_Legendre quadrature points, possibly clipped"""

    cell_sdf = sdf[cube]
    cell_alpha = 1.0 + FC_WEIGHT_SCALE * torch.tanh(weight[:, 12:20])

    reg_points, reg_weights = fem.geometry.element.Cube().instantiate_quadrature(
        order=order, family=fem.Polynomial.GAUSS_LEGENDRE
    )

    reg_qp = torch.tensor(reg_points, device="cuda", dtype=torch.float32)
    reg_qw = torch.tensor(reg_weights, device="cuda", dtype=torch.float32)

    n_qp = len(reg_qw)

    min_sdf, _ = torch.min(cell_sdf, dim=1)
    active_cells = torch.where(min_sdf < 0, 1, 0)

    qc = torch.zeros(size=(cell_sdf.shape[0], n_qp, 3), dtype=torch.float32, device="cuda")
    qw = torch.zeros(size=(cell_sdf.shape[0], n_qp), dtype=torch.float32, device="cuda")

    qc[:] = reg_qp
    qw[:] = reg_qw

    if clip:
        x = qc[:, :, 0]
        y = qc[:, :, 1]
        z = qc[:, :, 2]

        cell_s = (cell_sdf * cell_alpha).unsqueeze(-1)
        s = (
            (1.0 - x) * (1.0 - y) * (1.0 - z) * cell_s[:, 0]
            + (x) * (1.0 - y) * (1.0 - z) * cell_s[:, 1]
            + (1.0 - x) * (y) * (1.0 - z) * cell_s[:, 2]
            + (x) * (y) * (1.0 - z) * cell_s[:, 3]
            + (1.0 - x) * (1.0 - y) * (z) * cell_s[:, 4]
            + (x) * (1.0 - y) * (z) * cell_s[:, 5]
            + (1.0 - x) * (y) * (z) * cell_s[:, 6]
            + (x) * (y) * (z) * cell_s[:, 7]
        )

        qw *= torch.where(s <= 0.0, 1.0, 0.0)

    qc = qc.contiguous()
    qw = (qw + 1.0e-8).contiguous()

    return qc, qw, active_cells


@functools.cache
def _load_model(model_path: str):
    model = torch.jit.load(model_path)
    model.eval()
    return model


def get_quadrature(model_path, cube, sdf, weight, clip=True, order=0):
    sdf = torch.tensor(sdf, device="cuda")
    cube = torch.tensor(cube, device="cuda")
    weight = torch.tensor(weight, device="cuda")

    if model_path is None:
        qc, qw, active_cells = regular_quadrature(cube, sdf, weight, clip=clip, order=order)
    else:
        model = _load_model(model_path)
        qc, qw, active_cells = infer_quadrature(model, cube, sdf, weight)

    qc_wp = wp.clone(wp.from_torch(qc, dtype=wp.vec3, requires_grad=False))
    qw_wp = wp.clone(wp.from_torch(qw, dtype=wp.float32, requires_grad=False))
    active_cells = wp.clone(wp.from_torch(active_cells.int(), dtype=wp.int32, requires_grad=False))

    return qc_wp, qw_wp, active_cells


@fem.integrand
def surface_positions(s: fem.Sample, domain: fem.Domain, displacement: fem.Field):
    return domain(s) + displacement(s)


def sim_from_flexicubes(sim_class, flexi, geo: fem.Grid3D, args):
    """Instantiates a simulator instance from Flexicubes data"""

    fc_sdf = flexi["fc_sdf"]
    fc_pos = flexi["fc_pos"]
    fc_weights = flexi["fc_weights"]
    fc_stiff = flexi["fc_stiffness"]
    fc_cube = flexi["fc_cube"]

    res = geo.res[0]

    # Compute quadrature and active cells from flexicube sdf
    quad_model = None if args.reg_qp else args.quadrature_model
    quad_order = max(2 * args.degree, args.reg_qp)
    qc, qw, active_cells = get_quadrature(quad_model, fc_cube, fc_sdf, fc_weights, clip=args.clip, order=quad_order)

    # Create deformed grid
    grid_displacement_field = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3).make_field()
    grid_displacement_field.dof_values = fc_pos
    grid_displacement_field.dof_values.requires_grad = True

    deformed_grid = grid_displacement_field.make_deformed_geometry(relative=False)

    # Initialize sim
    sim = sim_class(deformed_grid, active_cells, args)

    sim.forces.count = 0
    sim.forces.centers = wp.zeros(
        shape=(1,),
        dtype=wp.vec3,
    )
    sim.forces.radii = wp.array([2.0 / res], dtype=float)
    sim.forces.forces = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3)

    sim.init_displacement_space()
    if fc_stiff is not None:
        sim.scale_lame_field(wp.array(fc_stiff, dtype=float))

    # Replace regular quadrature will learned quadrature
    domain = fem.Cells(sim._geo_partition)
    quadrature = fem.ExplicitQuadrature(domain, qc, qw)
    sim.vel_quadrature = quadrature
    sim.strain_quadrature = quadrature
    sim.elasticity_quadrature = quadrature

    # For Mixed FEM: locate strain nodes at quadrature points
    geo_quadrature = fem.ExplicitQuadrature(fem.Cells(deformed_grid), qc, qw)
    rbf_basis = fem.PointBasisSpace(geo_quadrature)
    sim.set_strain_basis(rbf_basis)
    sim.init_strain_spaces()

    return sim


def flexicubes_from_sdf_grid(res, sdf, pos, sdf_grad_func=None, output_tetmesh=False, device="cuda"):
    """Creates a Flexicubes datastructure from a SDF discretized on a dense grid"""

    fc = FlexiCubes(device)
    x_nx3, cube_fx8 = fc.construct_voxel_grid(res)

    surf_cubes, occ_fx8 = fc._identify_surf_cubes(sdf, cube_fx8)
    case_ids = fc._get_case_id(occ_fx8, surf_cubes, res)
    num_vd = torch.index_select(input=fc.num_vd_table, index=case_ids, dim=0)

    weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device=device)

    flexi = {
        "fc_pos": pos.detach().cpu().numpy(),
        "fc_sdf": sdf.detach().cpu().numpy(),
        "fc_cube": cube_fx8.detach().cpu().numpy(),
        "fc_weights": weight.detach().cpu().numpy(),
        "fc_stiffness": None,
        "fc_bd_cubes": surf_cubes.detach().cpu().numpy(),
        "fc_bd_nv": num_vd.detach().cpu().numpy(),
    }

    vertices, faces, L_dev = fc(
        voxelgrid_vertices=pos,
        scalar_field=sdf,
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
        flexi.update(
            {
                "tet_vertices": vertices.detach().cpu().numpy(),
                "tet_indices": faces.detach().cpu().numpy(),
                "vtx_displ": np.zeros(vertices.shape, dtype=np.float32),
            }
        )
    else:
        flexi.update(
            {
                "tri_vertices": vertices.detach().cpu().numpy(),
                "tri_faces": faces.detach().cpu().numpy(),
                "vtx_displ": np.zeros(vertices.shape, dtype=np.float32),
            }
        )

    return flexi
