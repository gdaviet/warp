# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest

import numpy as np

import warp as wp
import warp.sparse as wps
from warp.optim.linear import CG, CR, GMRES, BiCGSTAB, aslinearoperator, bicgstab, cg, cr, gmres, preconditioner
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


def _run_batched_gmres(test, device, dtype, batch_sizes, seed_base, tol, restart):
    rows = sum(batch_sizes)
    A_np_full = np.zeros((rows, rows), dtype=np.float64 if dtype == wp.float64 else np.float32)
    b_np_full = np.zeros(rows, dtype=A_np_full.dtype)

    for i, n in enumerate(batch_sizes):
        A_i, b_i = _make_nonsymmetric_system(n, seed=seed_base + i, dtype=dtype, device="cpu")
        offset = sum(batch_sizes[:i])
        sl = slice(offset, offset + n)
        A_np_full[sl, sl] = A_i.numpy()
        b_np_full[sl] = b_i.numpy()

    A_full = wp.array(A_np_full, dtype=dtype, device=device)
    b_full = wp.array(b_np_full, dtype=dtype, device=device)

    offsets = _batch_offsets(batch_sizes, device)
    A_op = aslinearoperator(A_full, batch_offsets=offsets)
    test.assertEqual(A_op.batch_count, len(batch_sizes))

    # Diagonal preconditioner (block-diagonal across the full system, so implicitly per-batch).
    M = preconditioner(A_full, "diag")

    # (description, kwargs) pairs — no precond, right precond, left precond
    cases = [
        ("none", {}),
        ("right", {"M": M}),
        ("left", {"M": M, "is_left_preconditioner": True}),
    ]
    for _label, kwargs in cases:
        x_full = wp.zeros_like(b_full)
        gmres(A_op, b_full, x_full, tol=tol, restart=restart, maxiter=1000, **kwargs)
        _check_batch_residuals(test, A_np_full, b_np_full, batch_sizes, x_full, tol, dtype)


def test_batched_gmres(test, device, dtype=wp.float32, batch_count=4, n=20):
    _run_batched_gmres(
        test,
        device,
        dtype,
        batch_sizes=[n] * batch_count,
        seed_base=456,
        tol=1e-3 if dtype == wp.float32 else 1e-5,
        restart=16,
    )


def test_batched_gmres_nonuniform(test, device, dtype=wp.float32):
    _run_batched_gmres(
        test,
        device,
        dtype,
        batch_sizes=[8, 15, 10, 12],
        seed_base=654,
        tol=1e-3 if dtype == wp.float32 else 1e-5,
        restart=16,
    )


def test_functor_reuse(test, device):
    # For each solver, construct a pre-allocated functor, then re-run on a different
    # (but compatible) system without re-allocating temporary buffers.
    cases = [
        (cg, CG, _make_spd_system, 32, {"maxiter": 500}),
        (cr, CR, _make_spd_system, 32, {"maxiter": 500}),
        (bicgstab, BiCGSTAB, _make_nonsymmetric_system, 32, {"maxiter": 500}),
        (gmres, GMRES, _make_nonsymmetric_system, 16, {"tol": 1.0e-3, "restart": 16, "maxiter": 256}),
    ]
    with wp.ScopedDevice(device):
        for func, klass, make_system, n, kwargs in cases:
            A1, b1 = make_system(n=n, seed=11, dtype=wp.float64, device=device)
            x1 = wp.zeros_like(b1)
            state = func(A1, b1, x1, run=False, **kwargs)
            test.assertIsInstance(state, klass)

            # First run with the original system
            _niter, err, atol = state()
            test.assertLessEqual(err, atol)

            # Second run with a *different* but compatible system
            A2, b2 = make_system(n=n, seed=22, dtype=wp.float64, device=device)
            x2 = wp.zeros_like(b2)
            _niter2, err2, atol2 = state(A=A2, b=b2, x=x2)
            test.assertLessEqual(err2, atol2)

            # Residual check in numpy to confirm x2 really solves A2 x2 = b2
            residual = A2.numpy() @ x2.numpy() - b2.numpy()
            test.assertLessEqual(np.linalg.norm(residual), 2.0 * atol2)


def test_functor_preconditioner(test, device):
    # CG and CR allow toggling M between None and a valid preconditioner between calls.
    with wp.ScopedDevice(device):
        A, b = _make_spd_system(n=32, seed=33, dtype=wp.float64, device=device)
        M = preconditioner(A, "diag")

        for func in (cg, cr):
            x = wp.zeros_like(b)
            state = func(A, b, x, maxiter=500, run=False)

            # No preconditioner on first call
            _, err, atol = state()
            test.assertLessEqual(err, atol)

            # With preconditioner on second call
            x.zero_()
            _, err2, atol2 = state(M=M)
            test.assertLessEqual(err2, atol2)


def test_functor_compat_errors(test, device):
    with wp.ScopedDevice(device):
        A, b = _make_spd_system(n=32, seed=44, dtype=wp.float64, device=device)
        x = wp.zeros_like(b)
        state = cg(A, b, x, maxiter=100, run=False)

        # Wrong b shape
        b_bad = wp.zeros(64, dtype=wp.float64, device=device)
        with test.assertRaises(ValueError):
            state(b=b_bad)

        # Wrong dtype
        A_bad, b_bad = _make_spd_system(n=32, seed=44, dtype=wp.float32, device=device)
        x_bad = wp.zeros_like(b_bad)
        with test.assertRaises(ValueError):
            state(A=A_bad, b=b_bad, x=x_bad)

        # BiCGSTAB requires M presence to match
        A2, b2 = _make_nonsymmetric_system(n=16, seed=45, dtype=wp.float64, device=device)
        x2 = wp.zeros_like(b2)
        M2 = preconditioner(A2, "diag")
        bic_state = bicgstab(A2, b2, x2, maxiter=100, run=False)  # M=None at construction
        with test.assertRaises(ValueError):
            bic_state(M=M2)


def _make_block_spd_bsr(n_blocks: int, block_size: int, seed: int, dtype, device):
    """Build an SPD block-dense BSR matrix with ``n_blocks`` by ``n_blocks`` blocks of
    size ``block_size`` by ``block_size``, plus a matching right-hand side."""
    rng = np.random.default_rng(seed)
    n = n_blocks * block_size
    C = rng.uniform(-1.0, 1.0, (n, n))
    A_dense = C @ C.T + np.eye(n) * n  # strongly SPD
    f = rng.uniform(-1.0, 1.0, n)
    b_np = A_dense @ f

    mat_type = wp.types.matrix(shape=(block_size, block_size), dtype=dtype)

    rows = np.repeat(np.arange(n_blocks), n_blocks).astype(np.int32)
    cols = np.tile(np.arange(n_blocks), n_blocks).astype(np.int32)
    blocks = np.empty((rows.size, block_size, block_size))
    for k, (br, bc) in enumerate(zip(rows, cols, strict=True)):
        blocks[k] = A_dense[br * block_size : (br + 1) * block_size, bc * block_size : (bc + 1) * block_size]

    A = wps.bsr_zeros(n_blocks, n_blocks, mat_type, device=device)
    wps.bsr_set_from_triplets(
        A,
        wp.array(rows, dtype=int, device=device),
        wp.array(cols, dtype=int, device=device),
        wp.array(blocks, dtype=mat_type, device=device),
    )

    np_dtype = np.float32 if dtype == wp.float32 else np.float64
    b = wp.array(b_np.astype(np_dtype), dtype=dtype, device=device)
    return A, b, A_dense.astype(np_dtype), b_np.astype(np_dtype)


def test_block_jacobi_preconditioner(test, device):
    for dtype in (wp.float32, wp.float64):
        A, b, A_dense, b_np = _make_block_spd_bsr(n_blocks=8, block_size=3, seed=12345, dtype=dtype, device=device)

        tol = 1e-3 if dtype == wp.float32 else 1e-8
        atol_scale = 32.0 if dtype == wp.float32 else 2.0

        for ptype in ("diag", "block_jacobi", "block_jacobi_ldlt"):
            M = preconditioner(A, ptype)
            x = wp.zeros(A.shape[0], dtype=dtype, device=device)
            with wp.ScopedDevice(A.device):
                _, err, atol = cg(A, b, x, M=M, maxiter=1000, tol=tol, use_cuda_graph=False)
            test.assertLessEqual(float(err), float(atol), msg=f"{ptype}/{dtype} did not converge")

            # Residual in numpy for independent verification.
            residual = A_dense @ x.numpy() - b_np
            test.assertLessEqual(
                np.linalg.norm(residual),
                atol_scale * float(atol),
                msg=f"{ptype}/{dtype} residual too large",
            )


def test_block_jacobi_input_errors(test, device):
    # Dense array input must be rejected.
    A_dense = wp.array(np.eye(8), dtype=wp.float64, device=device)
    with test.assertRaises(ValueError):
        preconditioner(A_dense, "block_jacobi")
    with test.assertRaises(ValueError):
        preconditioner(A_dense, "block_jacobi_ldlt")

    # Non-square block shape must be rejected.
    mat23 = wp.types.matrix(shape=(2, 3), dtype=wp.float64)
    A_rect = wps.bsr_zeros(3, 4, mat23, device=device)
    with test.assertRaises(ValueError):
        preconditioner(A_rect, "block_jacobi")
    with test.assertRaises(ValueError):
        preconditioner(A_rect, "block_jacobi_ldlt")


def test_block_jacobi_singular_block(test, device):
    # One zero diagonal block: the preconditioner must remain well-defined
    # (identity on the singular block row) rather than producing NaNs.
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wp.float64)
    rows = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=int, device=device)
    cols = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=int, device=device)
    blocks = np.stack([np.zeros((3, 3)), np.eye(3), np.eye(3)])
    vals = wp.array(blocks, dtype=mat33, device=device)

    A = wps.bsr_zeros(3, 3, mat33, device=device)
    wps.bsr_set_from_triplets(A, rows, cols, vals)

    x = wp.array(np.arange(1, 10, dtype=np.float64), dtype=wp.float64, device=device)
    for ptype in ("block_jacobi", "block_jacobi_ldlt"):
        M = preconditioner(A, ptype)
        z = wp.zeros_like(x)
        M.matvec(x, z, z, alpha=1.0, beta=0.0)
        z_np = z.numpy()
        test.assertTrue(np.all(np.isfinite(z_np)), msg=f"{ptype}: non-finite output on singular block")
        # Singular block row → identity applied → preserves x[:3]. Other rows → also identity here.
        np.testing.assert_allclose(z_np, x.numpy(), rtol=0, atol=0, err_msg=f"{ptype}")


def test_block_jacobi_scalar_fallback(test, device):
    # A CSR (1x1 block) input should fall back to the scalar diag path and
    # match the "diag" preconditioner bit-for-bit.
    n = 10
    rows = wp.array(np.arange(n, dtype=np.int32), dtype=int, device=device)
    cols = wp.array(np.arange(n, dtype=np.int32), dtype=int, device=device)
    vals = wp.array(np.full(n, 2.0, dtype=np.float64), dtype=wp.float64, device=device)

    A = wps.bsr_zeros(n, n, wp.float64, device=device)
    wps.bsr_set_from_triplets(A, rows, cols, vals)

    M_block = preconditioner(A, "block_jacobi")
    M_diag = preconditioner(A, "diag")

    x = wp.array(np.arange(1, n + 1, dtype=np.float64), dtype=wp.float64, device=device)
    z_block = wp.zeros_like(x)
    z_diag = wp.zeros_like(x)
    M_block.matvec(x, z_block, z_block, alpha=1.0, beta=0.0)
    M_diag.matvec(x, z_diag, z_diag, alpha=1.0, beta=0.0)
    np.testing.assert_allclose(z_block.numpy(), z_diag.numpy())


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
add_function_test(TestLinearSolvers, "test_batched_gmres_f32", test_batched_gmres, devices=devices)
add_function_test(TestLinearSolvers, "test_batched_gmres_f64", test_batched_gmres, devices=devices, dtype=wp.float64)
add_function_test(TestLinearSolvers, "test_batched_gmres_nonuniform", test_batched_gmres_nonuniform, devices=devices)
add_function_test(TestLinearSolvers, "test_batched_nonuniform", test_batched_nonuniform, devices=devices)
add_function_test(TestLinearSolvers, "test_functor_reuse", test_functor_reuse, devices=devices)
add_function_test(TestLinearSolvers, "test_functor_preconditioner", test_functor_preconditioner, devices=devices)
add_function_test(TestLinearSolvers, "test_functor_compat_errors", test_functor_compat_errors, devices=devices)
add_function_test(
    TestLinearSolvers, "test_block_jacobi_preconditioner", test_block_jacobi_preconditioner, devices=devices
)
add_function_test(TestLinearSolvers, "test_block_jacobi_input_errors", test_block_jacobi_input_errors, devices=devices)
add_function_test(
    TestLinearSolvers, "test_block_jacobi_singular_block", test_block_jacobi_singular_block, devices=devices
)
add_function_test(
    TestLinearSolvers, "test_block_jacobi_scalar_fallback", test_block_jacobi_scalar_fallback, devices=devices
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
