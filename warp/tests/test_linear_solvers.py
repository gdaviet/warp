# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest

import numpy as np

import warp as wp
from warp.optim.linear import aslinearoperator, bicgstab, cg, cr, gmres, preconditioner
from warp.tests.unittest_utils import *


def _check_linear_solve(test, A, b, func, *args, **kwargs):
    # test from zero
    x = wp.zeros_like(b)
    with wp.ScopedDevice(A.device):
        niter, err, atol = func(A, b, x, *args, use_cuda_graph=True, **kwargs)

    test.assertLessEqual(err, atol)

    # Test with capturable graph
    if A.device.is_cuda and wp.is_conditional_graph_supported():
        x.zero_()
        with wp.ScopedDevice(A.device):
            with wp.ScopedCapture() as capture:
                niter, err, atol = func(A, b, x, *args, use_cuda_graph=True, check_every=0, **kwargs)

            wp.capture_launch(capture.graph)

        niter = niter.numpy()[0]
        err = np.sqrt(err.numpy()[0])
        atol = np.sqrt(atol.numpy()[0])

        test.assertLessEqual(err, atol)

    # test with warm start
    with wp.ScopedDevice(A.device):
        niter_warm, err, atol = func(A, b, x, *args, use_cuda_graph=False, **kwargs)

    if isinstance(niter_warm, wp.array):
        niter_warm = niter_warm.numpy()[0]
        err = np.sqrt(err.numpy()[0])
        atol = np.sqrt(atol.numpy()[0])

    test.assertLessEqual(err, atol)

    if func in [cr, gmres]:
        # monotonic convergence
        test.assertLess(niter_warm, niter)

    # In CG and BiCGSTAB residual norm is evaluating from running residual
    # rather then being computed from scratch as Ax - b
    # This can lead to accumulated inaccuracies over iterations, esp in float32
    residual = A.numpy() @ x.numpy() - b.numpy()
    err_np = np.linalg.norm(residual)

    if A.dtype == wp.float64:
        test.assertLessEqual(err_np, 2.0 * atol)
    else:
        test.assertLessEqual(err_np, 32.0 * atol)


def _least_square_system(rng, n: int):
    C = rng.uniform(low=-100, high=100, size=(n, n))
    f = rng.uniform(low=-100, high=100, size=(n,))

    A = C @ C.T
    b = C @ f

    return A, b


def _make_spd_system(n: int, seed: int, dtype, device):
    rng = np.random.default_rng(seed)

    A, b = _least_square_system(rng, n)

    return wp.array(A, dtype=dtype, device=device), wp.array(b, dtype=dtype, device=device)


def _make_nonsymmetric_system(n: int, seed: int, dtype, device):
    rng = np.random.default_rng(seed)
    s = rng.uniform(low=0.1, high=10, size=(n,))

    A, b = _least_square_system(rng, n)
    A = A @ np.diag(s)

    return wp.array(A, dtype=dtype, device=device), wp.array(b, dtype=dtype, device=device)


def _make_indefinite_system(n: int, seed: int, dtype, device):
    rng = np.random.default_rng(seed)
    s = rng.uniform(low=0.1, high=10, size=(n,))

    A, b = _least_square_system(rng, n)
    A = A @ np.diag(s)

    return wp.array(A, dtype=dtype, device=device), wp.array(b, dtype=dtype, device=device)


def _make_identity_system(n: int, seed: int, dtype, device):
    rng = np.random.default_rng(seed)

    A = np.eye(n)
    b = rng.uniform(low=-1.0, high=1.0, size=(n,))

    return wp.array(A, dtype=dtype, device=device), wp.array(b, dtype=dtype, device=device)


def test_cg(test, device):
    A, b = _make_spd_system(n=64, seed=123, device=device, dtype=wp.float64)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, cg, maxiter=1000)
    _check_linear_solve(test, A, b, cg, M=M, maxiter=1000)

    A, b = _make_spd_system(n=16, seed=321, device=device, dtype=wp.float32)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, cg, maxiter=1000)
    _check_linear_solve(test, A, b, cg, M=M, maxiter=1000)

    A, b = _make_identity_system(n=5, seed=321, device=device, dtype=wp.float32)
    _check_linear_solve(test, A, b, cg, maxiter=30)


def test_cr(test, device):
    A, b = _make_spd_system(n=64, seed=123, device=device, dtype=wp.float64)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, cr, maxiter=1000)
    _check_linear_solve(test, A, b, cr, M=M, maxiter=1000)

    A, b = _make_spd_system(n=16, seed=321, device=device, dtype=wp.float32)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, cr, maxiter=1000)
    _check_linear_solve(test, A, b, cr, M=M, maxiter=1000)

    A, b = _make_identity_system(n=5, seed=321, device=device, dtype=wp.float32)
    _check_linear_solve(test, A, b, cr, maxiter=30)


def test_bicgstab(test, device):
    A, b = _make_nonsymmetric_system(n=64, seed=123, device=device, dtype=wp.float64)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, bicgstab, maxiter=1000)
    _check_linear_solve(test, A, b, bicgstab, M=M, maxiter=1000)
    _check_linear_solve(test, A, b, bicgstab, M=M, maxiter=1000, is_left_preconditioner=True)

    A, b = _make_nonsymmetric_system(n=16, seed=321, device=device, dtype=wp.float32)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, bicgstab, maxiter=1000)
    _check_linear_solve(test, A, b, bicgstab, M=M, maxiter=1000)
    _check_linear_solve(test, A, b, bicgstab, M=M, maxiter=1000, is_left_preconditioner=True)

    A, b = _make_indefinite_system(n=64, seed=121, device=device, dtype=wp.float64)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, bicgstab, maxiter=1000)
    _check_linear_solve(test, A, b, bicgstab, M=M, maxiter=1000)
    _check_linear_solve(test, A, b, bicgstab, M=M, maxiter=1000, is_left_preconditioner=True)

    A, b = _make_identity_system(n=5, seed=321, device=device, dtype=wp.float32)
    _check_linear_solve(test, A, b, bicgstab, maxiter=30)


def test_gmres(test, device):
    A, b = _make_nonsymmetric_system(n=64, seed=456, device=device, dtype=wp.float64)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, gmres, maxiter=1000, tol=1.0e-3)
    _check_linear_solve(test, A, b, gmres, M=M, maxiter=1000, tol=1.0e-5)
    _check_linear_solve(test, A, b, gmres, M=M, maxiter=1000, tol=1.0e-5, is_left_preconditioner=True)

    A, b = _make_nonsymmetric_system(n=64, seed=654, device=device, dtype=wp.float64)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, gmres, maxiter=1000, tol=1.0e-3)
    _check_linear_solve(test, A, b, gmres, M=M, maxiter=1000, tol=1.0e-5)
    _check_linear_solve(test, A, b, gmres, M=M, maxiter=1000, tol=1.0e-5, is_left_preconditioner=True)

    A, b = _make_identity_system(n=5, seed=123, device=device, dtype=wp.float32)
    _check_linear_solve(test, A, b, gmres, maxiter=120)


def _batch_offsets(batch_sizes, device):
    offsets = np.concatenate([[0], np.cumsum(batch_sizes)]).astype(np.int32)
    return wp.array(offsets, dtype=int, device=device)


def _check_batch_residuals(test, A_np_full, b_np_full, batch_sizes, x_full, tol, dtype):
    """Verify per-batch residuals match what _check_linear_solve uses."""
    scale = 32.0 if dtype == wp.float32 else 2.0
    x_np = x_full.numpy()
    offsets = np.concatenate([[0], np.cumsum(batch_sizes)])
    for i, (start, end) in enumerate(itertools.pairwise(offsets)):
        sl = slice(start, end)
        A_i = A_np_full[sl, sl].astype(np.float64)
        b_i = b_np_full[sl].astype(np.float64)
        res = np.linalg.norm(A_i @ x_np[sl].astype(np.float64) - b_i)
        atol_i = tol * np.linalg.norm(b_i)
        test.assertLessEqual(float(res), scale * atol_i, msg=f"batch {i}: residual {res:.2e} > {scale * atol_i:.2e}")


def test_batched_cg(test, device, dtype=wp.float32, batch_count=4, n=20):
    A_np_full = np.zeros((batch_count * n, batch_count * n), dtype=np.float64 if dtype == wp.float64 else np.float32)
    b_np_full = np.zeros(batch_count * n, dtype=A_np_full.dtype)

    for i in range(batch_count):
        A_i, b_i = _make_spd_system(n, seed=i, dtype=dtype, device="cpu")
        sl = slice(i * n, (i + 1) * n)
        A_np_full[sl, sl] = A_i.numpy()
        b_np_full[sl] = b_i.numpy()

    A_full = wp.array(A_np_full, dtype=dtype, device=device)
    b_full = wp.array(b_np_full, dtype=dtype, device=device)
    x_full = wp.zeros_like(b_full)

    offsets = _batch_offsets([n] * batch_count, device)
    A_op = aslinearoperator(A_full, batch_offsets=offsets)

    test.assertEqual(A_op.batch_count, batch_count)

    cg(A_op, b_full, x_full, tol=1e-5, maxiter=1000)

    _check_batch_residuals(test, A_np_full, b_np_full, [n] * batch_count, x_full, 1e-5, dtype)


def test_batched_cr(test, device, dtype=wp.float32, batch_count=4, n=20):
    A_np_full = np.zeros((batch_count * n, batch_count * n), dtype=np.float64 if dtype == wp.float64 else np.float32)
    b_np_full = np.zeros(batch_count * n, dtype=A_np_full.dtype)

    for i in range(batch_count):
        A_i, b_i = _make_spd_system(n, seed=i + 100, dtype=dtype, device="cpu")
        sl = slice(i * n, (i + 1) * n)
        A_np_full[sl, sl] = A_i.numpy()
        b_np_full[sl] = b_i.numpy()

    A_full = wp.array(A_np_full, dtype=dtype, device=device)
    b_full = wp.array(b_np_full, dtype=dtype, device=device)
    x_full = wp.zeros_like(b_full)

    offsets = _batch_offsets([n] * batch_count, device)
    A_op = aslinearoperator(A_full, batch_offsets=offsets)

    cr(A_op, b_full, x_full, tol=1e-5, maxiter=1000)

    _check_batch_residuals(test, A_np_full, b_np_full, [n] * batch_count, x_full, 1e-5, dtype)


def test_batched_bicgstab(test, device, dtype=wp.float32, batch_count=4, n=20):
    A_np_full = np.zeros((batch_count * n, batch_count * n), dtype=np.float64 if dtype == wp.float64 else np.float32)
    b_np_full = np.zeros(batch_count * n, dtype=A_np_full.dtype)

    for i in range(batch_count):
        A_i, b_i = _make_spd_system(n, seed=i + 200, dtype=dtype, device="cpu")
        sl = slice(i * n, (i + 1) * n)
        A_np_full[sl, sl] = A_i.numpy()
        b_np_full[sl] = b_i.numpy()

    A_full = wp.array(A_np_full, dtype=dtype, device=device)
    b_full = wp.array(b_np_full, dtype=dtype, device=device)
    x_full = wp.zeros_like(b_full)

    offsets = _batch_offsets([n] * batch_count, device)
    A_op = aslinearoperator(A_full, batch_offsets=offsets)

    bicgstab(A_op, b_full, x_full, tol=1e-5, maxiter=1000)

    _check_batch_residuals(test, A_np_full, b_np_full, [n] * batch_count, x_full, 1e-5, dtype)


def test_batched_nonuniform(test, device, dtype=wp.float32):
    batch_sizes = [8, 15, 10, 12]
    rows = sum(batch_sizes)
    A_np_full = np.zeros((rows, rows), dtype=np.float64 if dtype == wp.float64 else np.float32)
    b_np_full = np.zeros(rows, dtype=A_np_full.dtype)

    for i, n in enumerate(batch_sizes):
        A_i, b_i = _make_spd_system(n, seed=i + 300, dtype=dtype, device="cpu")
        offset = sum(batch_sizes[:i])
        sl = slice(offset, offset + n)
        A_np_full[sl, sl] = A_i.numpy()
        b_np_full[sl] = b_i.numpy()

    A_full = wp.array(A_np_full, dtype=dtype, device=device)
    b_full = wp.array(b_np_full, dtype=dtype, device=device)
    x_full = wp.zeros_like(b_full)

    offsets = _batch_offsets(batch_sizes, device)
    A_op = aslinearoperator(A_full, batch_offsets=offsets)
    test.assertEqual(A_op.batch_count, len(batch_sizes))

    cg(A_op, b_full, x_full, tol=1e-5, maxiter=1000)

    _check_batch_residuals(test, A_np_full, b_np_full, batch_sizes, x_full, 1e-5, dtype)


class TestLinearSolvers(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestLinearSolvers, "test_cg", test_cg, devices=devices)
add_function_test(TestLinearSolvers, "test_cr", test_cr, devices=devices)
add_function_test(TestLinearSolvers, "test_bicgstab", test_bicgstab, devices=devices)
add_function_test(TestLinearSolvers, "test_gmres", test_gmres, devices=devices)
add_function_test(TestLinearSolvers, "test_batched_cg_f32", test_batched_cg, devices=devices)
add_function_test(TestLinearSolvers, "test_batched_cg_f64", test_batched_cg, devices=devices, dtype=wp.float64)
add_function_test(TestLinearSolvers, "test_batched_cr_f32", test_batched_cr, devices=devices)
add_function_test(TestLinearSolvers, "test_batched_bicgstab_f32", test_batched_bicgstab, devices=devices)
add_function_test(TestLinearSolvers, "test_batched_nonuniform", test_batched_nonuniform, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
