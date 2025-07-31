#! /usr/bin/env uv run --script

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polyscope",
#     "warp-lang>=1.8.0",
# ]
# [tool.uv]
# find-links = ["https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html"]
# [tool.uv.sources]
# warp-lang = { index = "nvidia"}
# [[tool.uv.index]]
# name = "nvidia"
# url = "https://pypi.nvidia.com"
# explicit = true
# ///

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import argparse

from mfem.softbody_sim import ClassicFEM, run_softbody_sim

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem

# Demo app


@fem.integrand
def clamped_right(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    """Dirichlet boundary condition projector (fixed vertices selection)"""

    pos = domain(s)
    clamped = float(0.0)

    # clamped right sides
    clamped = wp.where(pos[0] >= 1.0, 1.0, 0.0)

    return wp.dot(u(s), v(s)) * clamped


if __name__ == "__main__":
    wp.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=10)
    ClassicFEM.add_parser_arguments(parser)
    args = parser.parse_args()

    pos, tets = fem_example_utils.gen_tetmesh(res=wp.vec3i(args.resolution), bounds_lo=wp.vec3(0.0, 0.75, 0.75))
    geo = fem.Tetmesh(positions=pos, tet_vertex_indices=tets)

    sim = ClassicFEM(geo, active_cells=None, args=args)
    sim.init_displacement_space()
    sim.init_strain_spaces()

    sim.set_boundary_condition(
        boundary_projector_form=clamped_right,
    )

    run_softbody_sim(sim)
