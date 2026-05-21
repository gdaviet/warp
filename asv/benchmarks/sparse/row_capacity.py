# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect

import numpy as np

import warp as wp
import warp.sparse as wps


def _require_row_capacity_support():
    probe = wps.bsr_zeros(0, 0, float)
    if not hasattr(probe, "row_ends"):
        raise NotImplementedError("Sparse row capacity support is not available")
    if not hasattr(wps, "bsr_compress"):
        raise NotImplementedError("bsr_compress is not available")
    if "topology" not in inspect.signature(wps.bsr_axpy).parameters:
        raise NotImplementedError("Sparse topology policies are not available")


def _make_row_capacity_matrix(nrow: int, ncol: int, active_per_row: int, capacity_per_row: int, device):
    offsets = np.arange(nrow + 1, dtype=np.int32) * capacity_per_row
    row_ends = offsets[:-1] + active_per_row

    columns = np.full(nrow * capacity_per_row, -1, dtype=np.int32)
    values = np.zeros(nrow * capacity_per_row, dtype=np.float32)

    row_beg = np.arange(nrow, dtype=np.int32)[:, None] % max(ncol - active_per_row, 1)
    active_cols = row_beg + np.arange(active_per_row, dtype=np.int32)[None, :]
    row_values = 1.0 / (1.0 + np.arange(active_per_row, dtype=np.float32))

    columns.reshape(nrow, capacity_per_row)[:, :active_per_row] = active_cols
    values.reshape(nrow, capacity_per_row)[:, :active_per_row] = row_values

    mat = wps.bsr_zeros(nrow, ncol, float, device=device)
    mat.nnz = columns.size
    mat.offsets = wp.array(offsets, dtype=int, device=device)
    mat.row_ends = wp.array(row_ends, dtype=int, device=device)
    mat.columns = wp.array(columns, dtype=int, device=device)
    mat.values = wp.array(values, dtype=float, device=device)
    return mat


def _make_duplicate_candidate_matrix(nrow: int, ncol: int, capacity_per_row: int, device):
    offsets = np.arange(nrow + 1, dtype=np.int32) * capacity_per_row
    row_ends = offsets[:-1] + capacity_per_row

    local_cols = np.array([0, 2, 1, 2, 3, 1, 4, 3], dtype=np.int32)
    local_vals = np.array([1.0, 2.0, 3.0, -0.5, 4.0, 1.0, 5.0, 0.25], dtype=np.float32)

    columns = np.empty(nrow * capacity_per_row, dtype=np.int32)
    values = np.empty(nrow * capacity_per_row, dtype=np.float32)

    row_beg = np.arange(nrow, dtype=np.int32)[:, None] % max(ncol - local_cols.max() - 1, 1)
    columns.reshape(nrow, capacity_per_row)[:, :] = row_beg + local_cols[None, :]
    values.reshape(nrow, capacity_per_row)[:, :] = local_vals[None, :]

    mat = wps.bsr_zeros(nrow, ncol, float, device=device)
    mat.nnz = columns.size
    mat.offsets = wp.array(offsets, dtype=int, device=device)
    mat.row_ends = wp.array(row_ends, dtype=int, device=device)
    mat.columns = wp.array(columns, dtype=int, device=device)
    mat.values = wp.array(values, dtype=float, device=device)
    return mat


class BsrMvGappedRows:
    """Test matrix-vector multiplication on a matrix with row-local slack capacity."""

    rounds = 1
    repeat = 2
    number = 20

    def setup(self):
        wp.init()
        _require_row_capacity_support()
        self.device = wp.get_device("cuda:0")

        with wp.ScopedDevice(self.device):
            self._mat = _make_row_capacity_matrix(32768, 32768, active_per_row=4, capacity_per_row=8, device=self.device)
            self._x = wp.ones(shape=self._mat.shape[1], dtype=wp.float32)
            self._y = wp.zeros(shape=self._mat.shape[0], dtype=wp.float32)
            self._mat.nnz_sync()
            self._run_impl()

        wp.synchronize_device(self.device)

    def _run_impl(self):
        wps.bsr_mv(self._mat, self._x, self._y, alpha=1.0, beta=0.0)

    def time_cuda(self):
        self._run_impl()
        wp.synchronize_device(self.device)


class BsrCompressGappedRows:
    """Test compact export from row-local candidate storage with duplicate columns."""

    rounds = 1
    repeat = 2
    number = 5

    def setup(self):
        wp.init()
        _require_row_capacity_support()
        self.device = wp.get_device("cuda:0")

        with wp.ScopedDevice(self.device):
            self._src = _make_duplicate_candidate_matrix(32768, 32768, capacity_per_row=8, device=self.device)
            self._dest = wps.bsr_zeros(self._src.nrow, self._src.ncol, float, device=self.device)
            self._run_impl()

        wp.synchronize_device(self.device)

    def _run_impl(self):
        wps.bsr_compress(self._src, dest=self._dest)

    def time_cuda(self):
        self._run_impl()
        wp.synchronize_device(self.device)


class BsrAxpyPaddedRows:
    """Test padded topology insertion into existing row capacity."""

    rounds = 1
    repeat = 2
    number = 5

    def setup(self):
        wp.init()
        _require_row_capacity_support()
        self.device = wp.get_device("cuda:0")

        with wp.ScopedDevice(self.device):
            self._x = _make_row_capacity_matrix(32768, 32768, active_per_row=4, capacity_per_row=4, device=self.device)
            self._y = _make_row_capacity_matrix(32768, 32768, active_per_row=0, capacity_per_row=8, device=self.device)
            self._run_impl()

        wp.synchronize_device(self.device)

    def _run_impl(self):
        wps.bsr_axpy(self._x, self._y, alpha=1.0, beta=0.0, topology="padded")

    def time_cuda(self):
        self._run_impl()
        wp.synchronize_device(self.device)
