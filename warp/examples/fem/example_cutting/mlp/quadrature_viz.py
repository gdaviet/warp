#! /usr/bin/env uv run --script

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kaolin==0.17.0",
#     "polyscope",
#     "torch==2.5.1",
#     "warp-lang",
#     "torchvision",
# ]
# [tool.uv]
# find-links = ["https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html"]
# ///

# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch
from kaolin.non_commercial import FlexiCubes


def quadrature(model, sdf):
    with torch.no_grad():
        qc, qw = model(torch.tensor(sdf, device="cuda", dtype=torch.float32).unsqueeze(0))

        return qc.squeeze(0).cpu().numpy(), qw.squeeze(0).cpu().numpy()


def fc_mesh(fc, res, x_nx3, sdf):
    pos = x_nx3 + 0.5
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    s = (
        (1.0 - x) * (1.0 - y) * (1.0 - z) * sdf[0]
        + (1.0 - x) * (1.0 - y) * (z) * sdf[1]
        + (1.0 - x) * (y) * (1.0 - z) * sdf[2]
        + (1.0 - x) * (y) * (z) * sdf[3]
        + (x) * (1.0 - y) * (1.0 - z) * sdf[4]
        + (x) * (1.0 - y) * (z) * sdf[5]
        + (x) * (y) * (1.0 - z) * sdf[6]
        + (x) * (y) * (z) * sdf[7]
    )

    with torch.no_grad():
        vertices, faces, L_dev = fc(
            pos,
            s,
            cube_fx8,
            res,
            beta=None,
            alpha=None,
            gamma_f=None,
            training=False,
        )

        return vertices.cpu().numpy(), faces.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("quadrature_model")
    args = parser.parse_args()

    model = torch.jit.load(args.quadrature_model)
    model.eval()

    fc = FlexiCubes("cpu")
    fc_res = 32
    x_nx3, cube_fx8 = fc.construct_voxel_grid(fc_res)

    vx, vy, vz = np.meshgrid(np.arange(2), np.arange(2), np.arange(2), indexing="ij")
    voxel = np.stack((vx.flatten(), vy.flatten(), vz.flatten()), axis=-1)

    voxel_egdes = np.array(
        [
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [4 + 0, 4 + 1],
            [4 + 1, 4 + 3],
            [4 + 3, 4 + 2],
            [4 + 2, 4 + 0],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=int,
    )

    sdf_faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [0, 4, 2, 6],
        [1, 5, 3, 7],
    ]
    face_names = [
        "X-",
        "X+",
        "Y-",
        "Y+",
        "Z-",
        "Z+",
    ]

    face_offsets = np.zeros(6)
    face_offsets[0] = -0.5
    face_offsets[1] = 0.5
    ps.init()

    ps.register_curve_network("voxel", voxel, voxel_egdes, radius=0.0025)

    vertex_offsets = np.zeros(8)

    def draw_elements():
        sdf = np.copy(vertex_offsets)
        for k, f in enumerate(sdf_faces):
            sdf[f] += face_offsets[k]

        qc, qw = quadrature(model, sdf)

        tri_vert, tri_faces = fc_mesh(fc, fc_res, x_nx3, sdf)

        pc = ps.register_point_cloud("qp", points=qc)
        pc.add_scalar_quantity("weight", qw * 0.25, enabled=True)
        pc.set_point_radius_quantity("weight", autoscale=False)

        if tri_faces.shape[0] > 0:
            sdf = ps.register_surface_mesh("sdf", vertices=tri_vert, faces=tri_faces, back_face_policy="custom")
        elif ps.has_surface_mesh("sdf"):
            ps.remove_surface_mesh("sdf")

    def frame_callback():
        any_changed = False
        for k, name in enumerate(face_names):
            changed, face_offsets[k] = psim.SliderFloat(f"Face {name}", face_offsets[k], v_min=-2, v_max=2)
            any_changed = any_changed or changed

        for k in range(vertex_offsets.shape[0]):
            changed, vertex_offsets[k] = psim.SliderFloat(
                f"Vertex ({k // 4}, {(k // 2) % 2}, {k % 2})",
                vertex_offsets[k],
                v_min=-2,
                v_max=2,
            )
            any_changed = any_changed or changed

        if any_changed:
            draw_elements()

        # ps.screenshot()

    draw_elements()

    ps.set_user_callback(frame_callback)
    ps.set_ground_plane_mode("none")
    ps.set_SSAA_factor(4)
    ps.set_build_default_gui_panels(False)

    # ps.get_surface_mesh("sdf").set_back_face_policy()

    ps.show()
