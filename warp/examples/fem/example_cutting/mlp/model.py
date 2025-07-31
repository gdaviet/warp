# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch import nn

import warp as wp
import warp.fem as fem


def lgl_basis(basis_dim: int):
    """Lobatto-Gauss-Legendre basis"""

    nodes, _weights = fem.polynomial.quadrature_1d(basis_dim, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    scales = fem.polynomial.lagrange_scales(nodes)

    return nodes, scales


def lagrange_weights(coords, lagrange_nodes, lagrange_scales):
    # Evaluate lagrange polynomials at coords

    coords = torch.clamp(coords, min=0.0, max=1.0)

    node_offsets = coords.unsqueeze(0) - lagrange_nodes
    basis_dim = len(lagrange_nodes)

    # annoyingly there's no exclusive cumprod, so we need separate exprs for ends
    node_prod = torch.cumprod(node_offsets, dim=0)
    node_rev_prod = torch.cumprod(node_offsets.flip(dims=(0,)), dim=0)

    lagrange = torch.empty_like(node_offsets)
    lagrange[0] = node_rev_prod[basis_dim - 2]
    for k in range(1, basis_dim - 1):
        lagrange[k] = node_prod[k - 1] * node_rev_prod[basis_dim - 1 - (k + 1)]
    lagrange[basis_dim - 1] = node_prod[basis_dim - 2]

    lagrange *= lagrange_scales
    return lagrange


class Quadrature(torch.nn.Module):
    """
    In: quadrature points and weights
        coords: batch x nqp x DIM
        W: batch x nqp
    Out: integratioon result for polynomial basis
        batch x BASIS_DIM ^ DIM
    """

    def __init__(self, device, basis_dim: int):
        super().__init__()

        self._BASIS_DIM = basis_dim

        lagrange_nodes, lagrange_scales = lgl_basis(basis_dim)

        self._lagrange_nodes = torch.tensor(
            lagrange_nodes,
            dtype=torch.float32,
            device=device,
        ).reshape((basis_dim, 1, 1, 1))
        self._lagrange_scales = torch.tensor(
            lagrange_scales,
            dtype=torch.float32,
            device=device,
        ).reshape((basis_dim, 1, 1, 1))

    def _eval_lagrange(self, coords):
        return lagrange_weights(coords, self._lagrange_nodes.detach(), self._lagrange_scales.detach())


class Quadrature2D(Quadrature):
    def forward(self, coords, W):
        BASIS_DIM = self._BASIS_DIM

        f = torch.empty((W.shape[0], BASIS_DIM, BASIS_DIM), device=W.device)

        lagrange = self._eval_lagrange(coords)

        for i in range(BASIS_DIM):
            for j in range(BASIS_DIM):
                f[:, i, j] = torch.sum(
                    W * lagrange[i, ..., 0] * lagrange[j, ..., 1],
                    dim=1,
                )

        return f


class Quadrature3D(Quadrature):
    def forward(self, coords, W):
        """
        In: quadrature points and weights
            coords: batch x nqp x DIM
            W: batch x nqp
        Out: integratioon result for polynomial basis
            batch x BASIS_DIM ^ DIM
        """

        BASIS_DIM = self._BASIS_DIM

        f = torch.empty((W.shape[0], BASIS_DIM, BASIS_DIM, BASIS_DIM), device=W.device)

        lagrange = self._eval_lagrange(coords)

        for i in range(BASIS_DIM):
            for j in range(BASIS_DIM):
                for k in range(BASIS_DIM):
                    f[:, i, j, k] = torch.sum(
                        W * lagrange[i, ..., 0] * lagrange[j, ..., 1] * lagrange[k, ..., 2],
                        dim=1,
                    )

        return f


class Shift(nn.Module):
    """Expresses quadrature points as shifts from reference Gauss-Legendre coords/weights"""

    def __init__(self, device, point_count, dim):
        super().__init__()

        # Gauss-Legendre coords and weights
        Q, W = fem.polynomial.quadrature_1d(point_count, family=fem.Polynomial.GAUSS_LEGENDRE)

        if dim == 2:
            Qk = np.array([[Q[i], Q[j]] for i, j in np.ndindex((point_count, point_count))])
            Wk = np.array([W[i] * W[j] for i, j in np.ndindex((point_count, point_count))])
        elif dim == 3:
            Qk = np.array([[Q[i], Q[j], Q[k]] for i, j, k in np.ndindex((point_count, point_count, point_count))])
            Wk = np.array([W[i] * W[j] * W[k] for i, j, k in np.ndindex((point_count, point_count, point_count))])

        self._DIM = dim
        self._Qk = torch.tensor(Qk, device=device, dtype=torch.float32)
        self._Wk = torch.tensor(Wk, device=device, dtype=torch.float32)

    def forward(self, x):
        DIM = self._DIM

        nqp = self._Wk.shape[0]
        x = x.reshape((-1, nqp, DIM + 1))

        Q = self._Qk.expand((x.shape[0], nqp, DIM)).detach()
        W = self._Wk.expand((x.shape[0], nqp)).detach()

        Q = Q + torch.tanh(x[..., :DIM])
        W = W * torch.exp(x[..., DIM])

        return Q, W


class Normalize(torch.nn.Module):
    """Normalize sdf input"""

    def forward(self, x):
        # normalize input

        center_grad = self._compute_center_grad(x)

        grad_norm = torch.linalg.vector_norm(center_grad, dim=0).unsqueeze(1)
        x_nor = x / (grad_norm + 1.0e-8)

        # clip full/empty cells
        x_min, _ = torch.min(x_nor, dim=1)
        x_max, _ = torch.max(x_nor, dim=1)

        empty = x_min > 1.0
        x_nor[empty] -= x_min[empty].unsqueeze(1) - 1.0

        full = x_max < -1.0
        x_nor[full] -= x_max[full].unsqueeze(1) + 1.0

        return x_nor


class Normalize3D(Normalize):
    def _compute_center_grad(self, x):
        # normalize input

        return 0.25 * torch.stack(
            (
                (x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3] - x[:, 4] - x[:, 5] - x[:, 6] - x[:, 7]),
                (x[:, 0] + x[:, 1] - x[:, 2] - x[:, 3] + x[:, 4] + x[:, 5] - x[:, 6] - x[:, 7]),
                (x[:, 0] - x[:, 1] + x[:, 2] - x[:, 3] + x[:, 4] - x[:, 5] + x[:, 6] - x[:, 7]),
            )
        )


class Normalize2D(Normalize):
    def _compute_center_grad(self, x):
        # normalize input
        return 0.5 * torch.stack(
            (
                (x[:, 0] + x[:, 1] - x[:, 2] - x[:, 3]),
                (x[:, 0] - x[:, 1] + x[:, 2] - x[:, 3]),
            )
        )


class PenLoss(torch.nn.Module):
    """Penalization of:
    - negative weights
    - points outside of cunit square/cube
    - ratio of min/max weight
    """

    def __init__(self, outside_coords_pen, conditioning_pen):
        super().__init__()

        self._OUTSIDE_COORDS_PEN = outside_coords_pen
        self._CONDITIONING_PEN = conditioning_pen

    def forward(self, coords, W):
        W_clamp = torch.clamp_min(W, min=1.0e-16)

        W_min, _ = torch.min(W_clamp, dim=1)
        W_max, _ = torch.max(W_clamp, dim=1)
        W_log_cond = torch.log(W_max) - torch.log(W_min)

        C_off = coords - torch.clamp(coords, min=0.0, max=1.0)

        pen_loss = self._OUTSIDE_COORDS_PEN * torch.linalg.vector_norm(
            C_off, ord=2
        ) + self._CONDITIONING_PEN * torch.sum(W_log_cond)

        return pen_loss


class Loss(torch.nn.Module):
    def __init__(self, device, dim, basis_dim, outside_coords_pen, conditioning_pen):
        super().__init__()

        self.pen_loss = PenLoss(outside_coords_pen, conditioning_pen)
        self.quad = Quadrature3D(device, basis_dim=basis_dim) if dim == 3 else Quadrature2D(device, basis_dim=basis_dim)
        self.loss = torch.nn.MSELoss(reduction="sum")

    def forward(self, coords, W, tgt):
        quad = self.quad(coords, W)
        return self.loss(quad, tgt) + self.pen_loss(coords, W)


class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, device, dim, point_count, width):
        super().__init__()

        qp_count = point_count**dim
        input_channels = 2**dim
        output_channels = qp_count * (dim + 1)

        self.layers = nn.Sequential(
            Normalize3D(),
            nn.Linear(input_channels, width, device=device),
            nn.ReLU(),
            nn.Linear(width, width, device=device),
            nn.ReLU(),
            nn.Linear(width, width, device=device),
            nn.ReLU(),
            nn.Linear(width, width, device=device),
            nn.ReLU(),
            nn.Linear(width, output_channels, device=device),
            Shift(device, point_count=point_count, dim=dim),
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


@wp.kernel(enable_backward=False)
def _gen_random_inputs(seed: int, ipt: wp.array2d(dtype=float)):
    i, k = wp.tid()
    state = wp.rand_init(123 + 33 * seed, i * ipt.shape[1] + k)

    s = wp.randn(state)

    while wp.abs(s) > 1.0e8:
        s = wp.randn(state)
    ipt[i, k] = s


def gen_random_inputs(inputs):
    seed = 0
    for ipt in inputs:
        wp_inputs_train = wp.from_torch(ipt)
        wp.launch(
            kernel=_gen_random_inputs,
            dim=ipt.shape,
            inputs=[seed, wp_inputs_train],
            device=wp_inputs_train.device,
        )
        seed += ipt.shape[0]


def gen_ground_truth(inputs, targets, dim, basis_dim, gt_res):
    BASIS_DIM = wp.constant(basis_dim)
    _lgl_nodes, _lgl_scales = lgl_basis(BASIS_DIM)
    _basis_vec = wp.vec(length=BASIS_DIM, dtype=float)
    LAGRANGE_NODES = wp.constant(_basis_vec(_lgl_nodes))
    LAGRANGE_SCALES = wp.constant(_basis_vec(_lgl_scales))

    @wp.func
    def lagrange_value(i: int, x: float):
        val = 1.0
        for k in range(BASIS_DIM):
            val *= wp.where(k == i, LAGRANGE_SCALES[i], x - LAGRANGE_NODES[k])

        return val

    @wp.kernel(enable_backward=False)
    def ground_truth_3d(ipt: wp.array2d(dtype=float), out: wp.array4d(dtype=float), gt_res: int):
        c, pi, pj, pk = wp.tid()

        res = wp.float64(0.0)

        h = 1.0 / float(gt_res)
        mass = h * h * h

        for i in range(gt_res):
            x = (0.5 + float(i)) * h
            lx = lagrange_value(pi, x)

            for j in range(gt_res):
                y = (0.5 + float(j)) * h
                ly = lagrange_value(pj, y)

                for k in range(gt_res):
                    z = (0.5 + float(k)) * h
                    lz = lagrange_value(pk, z)

                    s = (
                        (1.0 - x) * (1.0 - y) * (1.0 - z) * ipt[c, 0]
                        + (1.0 - x) * (1.0 - y) * (z) * ipt[c, 1]
                        + (1.0 - x) * (y) * (1.0 - z) * ipt[c, 2]
                        + (1.0 - x) * (y) * (z) * ipt[c, 3]
                        + (x) * (1.0 - y) * (1.0 - z) * ipt[c, 4]
                        + (x) * (1.0 - y) * (z) * ipt[c, 5]
                        + (x) * (y) * (1.0 - z) * ipt[c, 6]
                        + (x) * (y) * (z) * ipt[c, 7]
                    )
                    weight = wp.where(s > 0.0, 0.0, mass)

                    res += wp.float64(weight * lx * ly * lz)

        out[c, pi, pj, pk] = float(res)

    @wp.kernel(enable_backward=False)
    def ground_truth_2d(ipt: wp.array2d(dtype=float), out: wp.array3d(dtype=float), gt_res: int):
        c, pi, pj = wp.tid()

        res = wp.float64(0.0)

        h = 1.0 / float(gt_res)
        mass = h * h

        for i in range(gt_res):
            x = (0.5 + float(i)) * h
            lx = lagrange_value(pi, x)

            for j in range(gt_res):
                y = (0.5 + float(j)) * h
                ly = lagrange_value(pj, y)

                s = (
                    +(1.0 - x) * (1.0 - y) * ipt[c, 0]
                    + (1.0 - x) * (y) * ipt[c, 1]
                    + (x) * (1.0 - y) * ipt[c, 2]
                    + (x) * (y) * ipt[c, 3]
                )
                weight = wp.where(s > 0.0, 0.0, mass)

                res += wp.float64(weight * lx * ly)

        out[c, pi, pj] = float(res)

    ground_truth_kernel = ground_truth_3d if dim == 3 else ground_truth_2d

    for ipt, tgt in zip(inputs, targets):
        wp_inputs = wp.from_torch(ipt)
        wp_targets = wp.from_torch(tgt)

        wp.launch(
            kernel=ground_truth_kernel,
            dim=wp_targets.shape,
            inputs=[wp_inputs, wp_targets, gt_res],
            device=wp_inputs.device,
        )
