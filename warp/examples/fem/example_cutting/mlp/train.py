#! /usr/bin/env uv run --script

# /// script
# dependencies = [
#     "torch==2.5.1",
#     "warp-lang",
#     "matplotlib",
#     "pyqt5",
# ]
# ///

# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import argparse
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from model import MLP, Loss, gen_ground_truth, gen_random_inputs

import warp as wp

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dim", type=int, default=3, help="dimension of the space")
parser.add_argument("-o", "--order", type=int, default=2, help="order of the quadrature")
parser.add_argument("-i", "--iters", type=int, default=64000, help="number of iterations")
parser.add_argument("--n_train", type=int, default=24, help="number of training samples")
parser.add_argument("--n_test", type=int, default=10, help="number of test samples")
parser.add_argument("--n_batches", type=int, default=16, help="number of batches")
parser.add_argument("--conditioning_pen", type=float, default=0.00001, help="conditioning penalty")
parser.add_argument("--outside_coords_pen", type=float, default=10.0, help="outside coordinates penalty")
parser.add_argument("--gt_res", type=int, default=32, help="resolution of the ground truth brute-force integration")
parser.add_argument("--log_interval", type=int, default=100, help="number of iterations between residual logging")
parser.add_argument("--write_interval", type=int, default=1000, help="number of iterations between model dumps")
parser.add_argument("--device", type=str, default="cuda", help="device")


args = parser.parse_args()

DIM = args.dim
ORDER = args.order  # should be even

N_TRAIN = 2**args.n_train  # train samples
N_TEST = 2**args.n_test  # test size

N_BATCHES = args.n_batches  # number of batches

device = args.device

# dependent constants

POINTS = ORDER // 2 + 1

INPUTS = wp.constant(2**DIM)

MLP_WIDTH = 2 ** (2 + DIM + ORDER // 2)  # 64 for ORDER 2 works well, maybe 128 for order 4? 256 for order 6?


# lagrange polynomial basis or order ORDER used to evaluate quadrature accuracy
BASIS_DIM = wp.constant(ORDER + 1)  # number of polynomial coeffs in each axis of tensor product


print(f"Using {POINTS}^{DIM} quadrature points, {BASIS_DIM}^{DIM} basis polynomials, and MLP_WIDTH={MLP_WIDTH}")


# Construct our model by instantiating the class defined above
model = MLP(device=device, dim=DIM, point_count=POINTS, width=MLP_WIDTH)
model_scripted = torch.jit.script(model)  # Export to TorchScript

# Construct our loss function and an Optimizer.
criterion = Loss(
    device,
    dim=DIM,
    basis_dim=BASIS_DIM,
    outside_coords_pen=args.outside_coords_pen,
    conditioning_pen=args.conditioning_pen,
)
criterion_scripted = torch.jit.script(criterion)  # Export to TorchScript

# Create Tensors to hold input and outputs.

ipt_train = torch.empty((N_TRAIN, INPUTS), dtype=torch.float32, device=device)
ipt_test = torch.empty((N_TEST, INPUTS), dtype=torch.float32, device=device)

tgt_train = torch.empty((N_TRAIN, *((BASIS_DIM,) * DIM)), dtype=torch.float32, device=device)
tgt_test = torch.empty((N_TEST, *((BASIS_DIM,) * DIM)), dtype=torch.float32, device=device)

with wp.ScopedTimer(f"Generating {N_TRAIN} train + {N_TEST} test random cells...", synchronize=True):
    gen_random_inputs((ipt_train, ipt_test))

print("Generating ground truth...")
with wp.ScopedTimer("Generating ground truth", synchronize=True):
    gen_ground_truth(
        (ipt_train, ipt_test),
        (tgt_train, tgt_test),
        dim=DIM,
        basis_dim=BASIS_DIM,
        gt_res=args.gt_res,
    )

# Start training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)

start_time = time.monotonic()

for t in range(args.iters + 1):
    optimizer.zero_grad()

    bi = t % N_BATCHES
    bsize = N_TRAIN // N_BATCHES
    bslice = slice(bi * bsize, (bi + 1) * bsize)

    ipt_batch = ipt_train[bslice]
    tgt_batch = tgt_train[bslice]

    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model_scripted(ipt_batch)
    loss = criterion_scripted(*y_pred, tgt_batch)

    # Zero gradients, perform a backward pass, and update the weights.
    loss.backward()
    optimizer.step()
    scheduler.step()

    # print loss
    if t % args.log_interval == 0:
        loss_test = criterion_scripted(*model_scripted(ipt_test), tgt_test)
        time_elapsed = time.monotonic() - start_time
        print(f"#{t}\t train={loss.item()}\t test={loss_test.item()}\t time={time_elapsed}s")

    if t % args.write_interval == 0:
        model_scripted.save(f"quad_model_{DIM}d_o{ORDER}_{t}.pt")  # Save


model_scripted.save(f"quad_model_{DIM}d_o{ORDER}.pt")  # Save


# View a few results

N_VIZ = 16  # number of test samples for visualization

coords, W = model(ipt_test[:N_VIZ])


C_np = coords.cpu().detach().numpy()
W_np = W.cpu().detach().numpy()
sdf = ipt_test.cpu().numpy()

if DIM == 2:
    fig, axes = plt.subplots(N_VIZ // 4, 4)
    for k in range(N_VIZ):
        ax = axes[k // 4, k % 4]

        X, Y = np.meshgrid(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10))

        Cx = C_np[k, :, 0]
        Cy = C_np[k, :, 1]
        W = W_np[k]

        s = (
            (1.0 - X) * (1.0 - Y) * sdf[k, 0]
            + (1.0 - X) * (Y) * sdf[k, 1]
            + (X) * (1.0 - Y) * sdf[k, 2]
            + (X) * (Y) * sdf[k, 3]
        )

        H = np.heaviside(s, 1.0)
        ax.pcolormesh(X, Y, H)
        ax.scatter(Cx, Cy, s=W * 100.0)

        ax.scatter(
            np.dot(Cx, W) / np.sum(W),
            np.dot(Cy, W) / np.sum(W),
            s=100.0,
        )
else:
    fig, axes = plt.subplots(N_VIZ // 4, 4, subplot_kw={"projection": "3d"})
    for k in range(N_VIZ):
        ax = axes[k // 4, k % 4]

        X, Y, Z = np.meshgrid(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4))

        Cx = C_np[k, :, 0]
        Cy = C_np[k, :, 1]
        Cz = C_np[k, :, 2]
        W = W_np[k]

        s = (
            (1.0 - X) * (1.0 - Y) * (1.0 - Z) * sdf[k, 0]
            + (1.0 - X) * (1.0 - Y) * (Z) * sdf[k, 1]
            + (1.0 - X) * (Y) * (1.0 - Z) * sdf[k, 2]
            + (1.0 - X) * (Y) * (Z) * sdf[k, 3]
            + (X) * (1.0 - Y) * (1.0 - Z) * sdf[k, 4]
            + (X) * (1.0 - Y) * (Z) * sdf[k, 5]
            + (X) * (Y) * (1.0 - Z) * sdf[k, 6]
            + (X) * (Y) * (Z) * sdf[k, 7]
        )

        H = np.heaviside(s, 1.0)

        ax.scatter(X, Y, Z, c=H, s=0.2)

        ax.scatter(Cx, Cy, Cz, s=W * 100.0)

        ax.scatter(
            np.dot(Cx, W) / np.sum(W),
            np.dot(Cy, W) / np.sum(W),
            np.dot(Cz, W) / np.sum(W),
            s=100.0,
            c=1.0,
        )

plt.show()
