# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA smoke verification for sparse row-capacity support.

This script is intentionally separate from the normal unit tests so it can be
run by someone with a CUDA GPU while development is happening on CPU-only
machines. It exercises the capacity-aware paths that should be graph-capturable
when callers use ``overflow="ignore"`` and preallocated status/work arrays.
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import numpy as np

import warp as wp
import warp.sparse as wps


def _wp_int(values, device):
    return wp.array(np.asarray(values, dtype=np.int32), dtype=int, device=device)


def _wp_float(values, device):
    return wp.array(np.asarray(values, dtype=np.float32), dtype=float, device=device)


def _make_scalar_bsr(nrow, ncol, offsets, row_ends, columns, values, device):
    mat = wps.bsr_zeros(nrow, ncol, float, device=device)
    mat.nnz = len(columns)
    mat.offsets = _wp_int(offsets, device)
    mat.row_ends = _wp_int(row_ends, device)
    mat.columns = _wp_int(columns, device)
    mat.values = _wp_float(values, device)
    return mat


def _make_compact_scalar_bsr(nrow, ncol, rows, columns, values, device):
    return wps.bsr_from_triplets(
        nrow,
        ncol,
        rows=_wp_int(rows, device),
        columns=_wp_int(columns, device),
        values=_wp_float(values, device),
    )


def _make_mat22_source(device):
    mat = wps.bsr_zeros(1, 2, wp.mat22, device=device)
    mat.nnz = 3
    mat.offsets = _wp_int([0, 3], device)
    mat.row_ends = _wp_int([2], device)
    mat.columns = _wp_int([0, 1, -1], device)
    mat.values = wp.array(
        np.asarray(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=np.float32,
        ),
        dtype=wp.mat22,
        device=device,
    )
    return mat


def _dense_scalar(mat):
    offsets = mat.offsets.numpy()[: mat.nrow + 1]
    row_ends = mat.row_ends.numpy()[: mat.nrow]
    columns = mat.columns.numpy()[: mat.nnz]
    values = mat.values.numpy()[: mat.nnz]

    dense = np.zeros(mat.shape, dtype=np.float32)
    for row in range(mat.nrow):
        for block in range(int(offsets[row]), int(row_ends[row])):
            dense[row, int(columns[block])] += values[block]
    return dense


def _assert_array(name, actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if not np.array_equal(actual, expected):
        raise AssertionError(f"{name}: expected {expected.tolist()}, got {actual.tolist()}")


def _assert_dense(name, mat, expected):
    actual = _dense_scalar(mat)
    expected = np.asarray(expected, dtype=np.float32)
    if not np.allclose(actual, expected):
        raise AssertionError(f"{name}: expected\n{expected}\ngot\n{actual}")


def _assert_status(name, status):
    value = int(status.numpy()[0])
    if value != 0:
        raise AssertionError(f"{name}: expected status 0, got {value}")


def _assert_work_status(name, work):
    value = work.status_sync()
    if value != 0:
        raise AssertionError(f"{name}: expected status 0, got {value} ({work.status_message()})")


def _make_state(device):
    state = SimpleNamespace()

    state.a = _make_compact_scalar_bsr(
        2,
        3,
        rows=[0, 0, 1, 1],
        columns=[0, 2, 1, 2],
        values=[1.0, 2.0, 3.0, 4.0],
        device=device,
    )
    state.b = _make_compact_scalar_bsr(
        3,
        2,
        rows=[0, 1, 2],
        columns=[0, 1, 0],
        values=[5.0, 6.0, 7.0],
        device=device,
    )

    state.triplet_rows = _wp_int([0, 0, 0, 1, 1], device)
    state.triplet_cols = _wp_int([2, 0, 2, 1, 1], device)
    state.triplet_vals = _wp_float([1.0, 2.0, 3.0, 4.0, -4.0], device)
    state.triplet_dest = _make_scalar_bsr(2, 3, [0, 3, 6], [0, 3], [-1] * 6, [0.0] * 6, device)
    state.triplet_status = wp.zeros(1, dtype=int, device=device)

    state.assign_dest = _make_scalar_bsr(2, 3, [0, 3, 6], [0, 3], [-1] * 6, [0.0] * 6, device)
    state.assign_status = wp.zeros(1, dtype=int, device=device)

    state.block_src = _make_mat22_source(device)
    state.reblock_dest = _make_scalar_bsr(2, 4, [0, 6, 12], [0, 6], [-1] * 12, [0.0] * 12, device)
    state.reblock_status = wp.zeros(1, dtype=int, device=device)

    state.axpy_y = _make_scalar_bsr(
        2,
        3,
        [0, 3, 6],
        [1, 4],
        [1, -1, -1, 0, -1, -1],
        [10.0, 0.0, 0.0, 20.0, 0.0, 0.0],
        device,
    )
    state.axpy_work = wps.bsr_axpy_work_arrays()

    state.transpose_dest = _make_scalar_bsr(3, 2, [0, 2, 4, 6], [0, 2, 4], [-1] * 6, [0.0] * 6, device)
    state.transpose_status = wp.zeros(1, dtype=int, device=device)

    state.mm_dest = _make_scalar_bsr(2, 2, [0, 2, 4], [0, 2], [-1] * 4, [0.0] * 4, device)
    state.mm_work = wps.bsr_mm_work_arrays()

    state.right_diag = _make_compact_scalar_bsr(2, 2, [0, 1], [0, 1], [3.0, 4.0], device)
    state.alias_x = _make_scalar_bsr(2, 2, [0, 2, 4], [1, 3], [0, -1, 1, -1], [1.0, 0.0, 2.0, 0.0], device)
    state.alias_work = wps.bsr_mm_work_arrays()

    return state


def _run_body(state):
    wps.bsr_set_from_triplets(
        state.triplet_dest,
        state.triplet_rows,
        state.triplet_cols,
        state.triplet_vals,
        topology="padded",
        overflow="ignore",
        status=state.triplet_status,
    )
    wps.bsr_assign(
        state.assign_dest,
        state.a,
        topology="padded",
        overflow="ignore",
        status=state.assign_status,
    )
    wps.bsr_assign(
        state.reblock_dest,
        state.block_src,
        topology="padded",
        overflow="ignore",
        status=state.reblock_status,
    )
    wps.bsr_axpy(
        state.a,
        state.axpy_y,
        alpha=2.0,
        beta=3.0,
        topology="padded",
        overflow="ignore",
        work_arrays=state.axpy_work,
    )
    wps.bsr_set_transpose(
        state.transpose_dest,
        state.a,
        topology="padded",
        overflow="ignore",
        status=state.transpose_status,
    )
    wps.bsr_mm(
        state.a,
        state.b,
        state.mm_dest,
        topology="padded",
        overflow="ignore",
        work_arrays=state.mm_work,
    )
    wps.bsr_mm(
        state.alias_x,
        state.right_diag,
        state.alias_x,
        topology="padded",
        overflow="ignore",
        work_arrays=state.alias_work,
    )


def _verify_state(state):
    _assert_status("bsr_set_from_triplets padded", state.triplet_status)
    _assert_status("bsr_assign padded", state.assign_status)
    _assert_status("bsr_assign padded reblock", state.reblock_status)
    _assert_work_status("bsr_axpy padded", state.axpy_work)
    _assert_status("bsr_set_transpose padded", state.transpose_status)
    _assert_work_status("bsr_mm padded", state.mm_work)
    _assert_work_status("bsr_mm padded aliased", state.alias_work)

    _assert_array("triplet row_ends", state.triplet_dest.row_ends.numpy(), [2, 3])
    _assert_array("triplet columns", state.triplet_dest.columns.numpy(), [0, 2, -1, -1, -1, -1])
    _assert_dense("triplet dense", state.triplet_dest, [[2.0, 0.0, 4.0], [0.0, 0.0, 0.0]])

    _assert_array("assign row_ends", state.assign_dest.row_ends.numpy(), [2, 5])
    _assert_array("assign columns", state.assign_dest.columns.numpy(), [0, 2, -1, 1, 2, -1])
    _assert_dense("assign dense", state.assign_dest, [[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]])

    _assert_array("reblock row_ends", state.reblock_dest.row_ends.numpy(), [4, 10])
    _assert_array("reblock columns", state.reblock_dest.columns.numpy(), [0, 1, 2, 3, -1, -1, 0, 1, 2, 3, -1, -1])
    _assert_dense("reblock dense", state.reblock_dest, [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]])

    _assert_array("axpy row_ends", state.axpy_y.row_ends.numpy(), [3, 6])
    _assert_array("axpy columns", state.axpy_y.columns.numpy(), [0, 1, 2, 0, 1, 2])
    _assert_dense("axpy dense", state.axpy_y, [[2.0, 30.0, 4.0], [60.0, 6.0, 8.0]])

    _assert_array("transpose row_ends", state.transpose_dest.row_ends.numpy(), [1, 3, 6])
    _assert_array("transpose columns", state.transpose_dest.columns.numpy(), [0, -1, 1, -1, 0, 1])
    _assert_dense("transpose dense", state.transpose_dest, [[1.0, 0.0], [0.0, 3.0], [2.0, 4.0]])

    _assert_array("mm row_ends", state.mm_dest.row_ends.numpy(), [1, 4])
    _assert_array("mm columns", state.mm_dest.columns.numpy(), [0, -1, 0, 1])
    _assert_dense("mm dense", state.mm_dest, [[19.0, 0.0], [28.0, 18.0]])

    _assert_array("aliased mm row_ends", state.alias_x.row_ends.numpy(), [1, 3])
    _assert_array("aliased mm columns", state.alias_x.columns.numpy(), [0, -1, 1, -1])
    _assert_dense("aliased mm dense", state.alias_x, [[4.0, 0.0], [0.0, 10.0]])

    reblocked_copy = wps.bsr_copy(state.block_src, block_shape=(1, 1), topology="padded")
    _assert_array("reblocked copy offsets", reblocked_copy.offsets.numpy(), [0, 6, 12])
    _assert_array("reblocked copy row_ends", reblocked_copy.row_ends.numpy(), [4, 10])
    _assert_array("reblocked copy columns", reblocked_copy.columns.numpy(), [0, 1, 2, 3, -1, -1, 0, 1, 2, 3, -1, -1])
    _assert_dense("reblocked copy dense", reblocked_copy, [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="CUDA device alias to verify, e.g. cuda:0")
    args = parser.parse_args(argv)

    wp.init()

    if not wp.is_cuda_available():
        raise SystemExit("CUDA is not available in this Warp build")

    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise SystemExit(f"Expected a CUDA device, got {device}")

    try:
        wp.set_mempool_enabled(device, True)
    except Exception as exc:
        raise SystemExit(f"CUDA graph capture requires mempool support on {device}: {exc}") from exc

    with wp.ScopedDevice(device):
        warmup = _make_state(device)
        _run_body(warmup)
        wp.synchronize_device(device)

        state = _make_state(device)
        with wp.ScopedCapture(force_module_load=False) as capture:
            _run_body(state)

        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

    _verify_state(state)
    print(f"sparse row-capacity CUDA graph smoke passed on {device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
