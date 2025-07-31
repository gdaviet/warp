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


import functools

import warp as wp
import warp.fem as fem

__all__ = ["neural_quadrature", "regular_quadrature"]


@wp.kernel
def instantiate_regular_quadrature(
    clip: bool,
    cell_indices: wp.array2d(dtype=int),
    vertex_sdf: wp.array(dtype=float),
    cell_alphas: wp.array2d(dtype=float),
    regular_coords: wp.array(dtype=wp.vec3),
    regular_weights: wp.array(dtype=float),
    cell_qp_coords: wp.array2d(dtype=wp.vec3),
    cell_qp_weights: wp.array2d(dtype=float),
    active_cells: wp.array(dtype=int),
):
    """Fill per-cell quadrature points, clipping exterior ones if requested,
    and mark inactive cells"""

    i = wp.tid()
    vidx = cell_indices[i]
    alphas = cell_alphas[i]

    # test if active
    min_sdf = float(1.0e8)
    for k in range(cell_indices.shape[1]):
        min_sdf = wp.min(min_sdf, vertex_sdf[vidx[k]])
    active_cells[i] = wp.where(min_sdf <= 0.0, 1, 0)

    for j in range(regular_coords.shape[0]):
        coords = regular_coords[j]
        cell_qp_coords[i, j] = coords
        cell_qp_weights[i, j] = regular_weights[j]

        if clip:
            # Clip quadrature -- disable exterior qps

            x = coords[0]
            y = coords[1]
            sdf = (
                (1.0 - x) * (1.0 - y) * vertex_sdf[vidx[0]] * alphas[0]
                + (x) * (1.0 - y) * vertex_sdf[vidx[1]] * alphas[1]
                + (1.0 - x) * (y) * vertex_sdf[vidx[2]] * alphas[2]
                + (x) * (y) * vertex_sdf[vidx[3]] * alphas[3]
            )

            if cell_indices.shape[1] > 4:
                z = coords[2]
                sdf = (1.0 - z) * sdf + z * (
                    (1.0 - x) * (1.0 - y) * vertex_sdf[vidx[4]] * alphas[4]
                    + (x) * (1.0 - y) * vertex_sdf[vidx[5]] * alphas[5]
                    + (1.0 - x) * (y) * vertex_sdf[vidx[6]] * alphas[6]
                    + (x) * (y) * vertex_sdf[vidx[7]] * alphas[7]
                )

            if sdf > 0.0:
                cell_qp_weights[i, j] = 0.0


def regular_quadrature(cell_vtx, sdf, cell_alpha, clip=True, order=2):
    """Regular Gauss_Legendre quadrature points, possibly clipped"""

    cell_vtx = wp.array(cell_vtx, dtype=int)
    sdf = wp.array(sdf, dtype=float)
    cell_alpha = wp.array(cell_alpha, dtype=float)

    if cell_vtx.shape[1] == 8:
        reg_points, reg_weights = fem.geometry.element.Cube().instantiate_quadrature(
            order=order, family=fem.Polynomial.GAUSS_LEGENDRE
        )
    else:
        reg_points, reg_weights = fem.geometry.element.Square().instantiate_quadrature(
            order=order, family=fem.Polynomial.GAUSS_LEGENDRE
        )

    n_qp = len(reg_weights)
    reg_qp = wp.array(reg_points, dtype=wp.vec3)
    reg_qw = wp.array(reg_weights, dtype=float)

    qc = wp.empty(shape=(cell_vtx.shape[0], n_qp), dtype=wp.vec3)
    qw = wp.empty(shape=(cell_vtx.shape[0], n_qp), dtype=float)
    active_cells = wp.empty(shape=(cell_vtx.shape[0]), dtype=int)

    wp.launch(
        instantiate_regular_quadrature,
        dim=cell_vtx.shape[0],
        inputs=[clip, cell_vtx, sdf, cell_alpha, reg_qp, reg_qw],
        outputs=[qc, qw, active_cells],
    )

    return qc, qw, active_cells


@functools.cache
def _load_model(model_path: str):
    import torch

    model = torch.jit.load(model_path)
    model.eval()
    return model


def infer_quadrature(model, cell_vtx, sdf, cell_alpha):
    """Inferred quadrature points from MLP"""
    import torch

    cell_sdf = sdf[cell_vtx]

    qc, qw = model(cell_sdf * cell_alpha)

    qc = qc.float().flip(dims=(2,))  # flip because FC cube corners are z-major

    min_sdf, _ = torch.min(cell_sdf, dim=1)
    active_cells = torch.where(min_sdf < 0, 1, 0)

    if qc.shape[-1] == 2:
        qc = torch.cat((qc, torch.zeros_like(qc[..., -1]).unsqueeze(-1)), dim=2)

    qc = qc.contiguous()
    qw = (qw + 1.0e-8).contiguous()

    return qc, qw, active_cells


def neural_quadrature(model_path, cell_vtx, sdf, cell_alpha):
    import torch

    # convert to torch and run inference
    device = str(wp.get_device())

    sdf = torch.tensor(sdf, device=device, dtype=torch.float32)
    cube = torch.tensor(cell_vtx, device=device, dtype=torch.int)
    cell_alpha = torch.tensor(cell_alpha, device=device, dtype=torch.float32)

    model = _load_model(model_path)
    qc, qw, active_cells = infer_quadrature(model, cube, sdf, cell_alpha)

    qc_wp = wp.clone(wp.from_torch(qc, dtype=wp.vec3, requires_grad=False))
    qw_wp = wp.clone(wp.from_torch(qw, dtype=wp.float32, requires_grad=False))
    active_cells = wp.clone(wp.from_torch(active_cells.int(), dtype=wp.int32, requires_grad=False))

    return qc_wp, qw_wp, active_cells
