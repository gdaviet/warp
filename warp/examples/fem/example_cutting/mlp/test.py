#! /usr/bin/env uv run --script

# /// script
# dependencies = [
#     "torch==2.5.1",
#     "warp-lang",
#     "matplotlib",
#     "pyqt5",
#     "scienceplots",
# ]
# ///

# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import argparse
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rc
from model import Loss, gen_ground_truth, lagrange_weights, lgl_basis

import warp as wp
import warp.fem as fem


def _regular_quadrature(ipt, order, clip):
    reg_points, reg_weights = fem.geometry.element.Cube().instantiate_quadrature(
        order=order, family=fem.Polynomial.GAUSS_LEGENDRE
    )
    reg_qp = torch.tensor(reg_points, device="cuda", dtype=torch.float32)
    reg_qw = torch.tensor(reg_weights, device="cuda", dtype=torch.float32)
    n_qp = len(reg_qw)

    qc = torch.zeros(size=(ipt.shape[0], n_qp, 3), dtype=torch.float32, device="cuda")
    qw = torch.zeros(size=(ipt.shape[0], n_qp), dtype=torch.float32, device="cuda")

    qc[:] = reg_qp
    qw[:] = reg_qw

    if clip:
        x = qc[:, :, 0]
        y = qc[:, :, 1]
        z = qc[:, :, 2]

        print(x.shape)
        ipt = ipt[:, :]

        s = (
            (1.0 - x) * (1.0 - y) * (1.0 - z) * ipt[:, np.newaxis, 0]
            + (1.0 - x) * (1.0 - y) * (z) * ipt[:, np.newaxis, 1]
            + (1.0 - x) * (y) * (1.0 - z) * ipt[:, np.newaxis, 2]
            + (1.0 - x) * (y) * (z) * ipt[:, np.newaxis, 3]
            + (x) * (1.0 - y) * (1.0 - z) * ipt[:, np.newaxis, 4]
            + (x) * (1.0 - y) * (z) * ipt[:, np.newaxis, 5]
            + (x) * (y) * (1.0 - z) * ipt[:, np.newaxis, 6]
            + (x) * (y) * (z) * ipt[:, np.newaxis, 7]
        )

        qw *= torch.where(s <= 0.0, 1.0, 0.0)

    return qc, qw


def muller_quadrature(ipt, dim, order, gt_res):
    basis_dim = order + 1
    lgl_nodes, lgl_scale = lgl_basis(basis_dim=basis_dim)

    reg_points, reg_weights = fem.geometry.element.Cube().instantiate_quadrature(
        order=order, family=fem.Polynomial.GAUSS_LEGENDRE
    )

    device = ipt.device

    qp = torch.tensor(reg_points, device=device).unsqueeze(0)

    L = lagrange_weights(
        qp,
        torch.tensor(lgl_nodes, device=device).reshape(-1, 1, 1, 1),
        torch.tensor(lgl_scale, device=device).reshape(-1, 1, 1, 1),
    )

    Q = torch.empty((qp.shape[1], basis_dim, basis_dim, basis_dim), device=device)

    for i in range(basis_dim):
        for j in range(basis_dim):
            for k in range(basis_dim):
                Q[:, i, j, k] = L[i, ..., 0] * L[j, ..., 1] * L[k, ..., 2]

    Q = Q.reshape(-1, basis_dim**3)
    QQt = Q @ Q.T
    QQti = torch.linalg.inv(QQt)

    Q_dagger = Q.T @ QQti

    # target moment (ground truth)
    targets = torch.empty((ipt.shape[0], *((basis_dim,) * dim)), dtype=torch.float32, device=device)

    gen_ground_truth(
        (ipt,),
        (targets,),
        dim=dim,
        basis_dim=basis_dim,
        gt_res=gt_res,
    )

    W = targets.reshape(-1, basis_dim**dim) @ Q_dagger

    qp = qp.broadcast_to(ipt.shape[0], qp.shape[1], 3)

    return qp, W


def test_model(model_path, order, args):
    dim = args.dim
    device = args.device
    n_test = 2**args.n_test  # test size

    basis_dim = order + 1

    # Construct our model by instantiating the class defined above

    if model_path == "clip":
        model = partial(_regular_quadrature, order=order, clip=True)
    elif model_path == "full":
        model = partial(_regular_quadrature, order=order, clip=False)
    elif model_path == "muller":
        model = partial(muller_quadrature, order=order, dim=dim, gt_res=args.gt_res)
    else:
        model = torch.jit.load(model_path)
        model.eval()

    # Construct our loss function and an Optimizer.
    loss_module = Loss(
        device,
        dim=dim,
        basis_dim=basis_dim,
        outside_coords_pen=0.0,
        conditioning_pen=0.0,
    )

    # Use uniform distribution (training was done with normal distribution, but we hope to be robust)
    ipt_test = torch.rand((n_test, 2**dim), dtype=torch.float32, device=device) * 4.0 - 2.0
    tgt_test = torch.empty((n_test, *((basis_dim,) * dim)), dtype=torch.float32, device=device)

    print(f"Generating ground truth for {model_path}...")
    with wp.ScopedTimer("Generating ground truth", synchronize=True):
        gen_ground_truth(
            (ipt_test,),
            (tgt_test,),
            dim=dim,
            basis_dim=basis_dim,
            gt_res=args.gt_res,
        )

    with wp.ScopedTimer("Computing weights", synchronize=True):
        qp_coords, qp_weights = model(ipt_test)

    quad_pred = loss_module.quad(qp_coords, qp_weights)
    # quad_pred = torch.zeros_like(tgt_test)

    error = (quad_pred - tgt_test).flatten().abs().detach().cpu().numpy()

    W_clamp = torch.clamp_min(qp_weights, min=1.0e-16).reshape(qp_weights.shape[0], -1)
    W_min, _ = torch.min(W_clamp, dim=1)
    W_max, _ = torch.max(W_clamp, dim=1)
    W_cond = W_max / W_min

    return error, W_cond.detach().cpu().numpy()


parser = argparse.ArgumentParser()

parser.add_argument(
    "quadrature_models",
    nargs="+",
    help="path to the quadrature models to compare, or predefined formula (clip, full, muller)",
)
parser.add_argument("-d", "--dim", type=int, default=3, help="dimension of the space")
parser.add_argument("-o", "--orders", type=int, default=2, nargs="+", help="orders of the quadrature")
parser.add_argument("-c", "--cond", type=int, nargs="+", help="conditioning penalty (for plot labels)")
parser.add_argument(
    "-l",
    "--labels",
    type=str,
    nargs="+",
    help="labels for the quadrature models (for plot labels)",
)
parser.add_argument("--n_test", type=int, default=10, help="number of test samples")
parser.add_argument(
    "--gt_res",
    type=int,
    default=32,
    help="resolution of the ground truth brute-force integration",
)
parser.add_argument("--device", type=str, default="cuda", help="device")

args = parser.parse_args()

if isinstance(args.orders, int):
    args.orders = [args.orders] * len(args.quadrature_models)

errors, conditions = zip(
    *(test_model(model_path, order, args) for model_path, order in zip(args.quadrature_models, args.orders))
)

if args.labels:
    labels = [
        f"$\\begin{{array}}{{c}}\\text{{{ll}}}\\\\d={o}\\end{{array}}$" for ll, o in zip(args.labels, args.orders)
    ]
elif args.cond:
    labels = [
        f"$\\begin{{array}}{{c}}d={o}\\\\\\gamma_\\star=10^{{-{p}}}\\end{{array}}$"
        for p, o in zip(args.cond, args.orders)
    ]
else:
    labels = args.quadrature_models


rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)


plt.style.use(["science", "ieee"])

plt.figure(figsize=(8 / 2.54, 4 / 2.54))
plt.boxplot(errors, labels=labels, whis=1.5)
plt.yscale("log")
plt.ylabel(r"Integration error $\mathcal{Q}_K$")
plt.ylim(1e-5, 0.1)
plt.savefig("test_error.pdf")
plt.show()


plt.figure(figsize=(8 / 2.54, 4 / 2.54))
plt.boxplot(conditions, labels=labels, whis=1.5)
plt.yscale("log")
plt.ylabel("Condition number")
plt.ylim(1.0, 1.0e3)
plt.savefig("test_cond.pdf")
plt.show()
