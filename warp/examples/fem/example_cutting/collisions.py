# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
from typing import Any, List

from softbody_sim import SoftbodySim

import warp as wp
import warp.fem as fem
import warp.sparse as sp


class CollisionHandler:
    @staticmethod
    def add_parser_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--collision_stiffness",
            "-ck",
            type=float,
            default=1.0,
            help="Multiplier for collision force/energy",
        )
        parser.add_argument(
            "--collision_radius",
            "-cr",
            type=float,
            default=0.1,
            help="Radius of interaction for collision particles",
        )
        parser.add_argument(
            "--collision_detection_ratio",
            "-cd",
            type=float,
            default=2.0,
            help="Multiplier of collision radius for detection",
        )
        parser.add_argument("--friction", "-mu", type=float, default=0.2)
        parser.add_argument(
            "--friction_reg",
            "-mur",
            type=float,
            default=0.1,
            help="Regularization coefficient for friction",
        )
        parser.add_argument(
            "--friction_fluid",
            "-nu",
            type=float,
            default=0.01,
            help="Additional fluid friction ratio to convexify things",
        )
        parser.add_argument(
            "--ground",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Do ground collisions",
        )
        parser.add_argument(
            "--ground_height",
            type=float,
            default=0.0,
            help="Ground height",
        )

    def __init__(
        self,
        kinematic_meshes: List[wp.Mesh],
        cp_cell_indices,
        cp_cell_coords,
        sim: SoftbodySim,
    ):
        self.args = sim.args
        self.sim = sim
        self.warp_meshes = kinematic_meshes

        n_cp = cp_cell_indices.shape[0]
        collision_quadrature = fem.PicQuadrature(
            domain=sim.vel_quadrature.domain,
            positions=(cp_cell_indices, cp_cell_coords),
            measures=wp.ones(n_cp, dtype=float),
        )
        self.set_collision_quadrature(collision_quadrature)

        self.n_contact = 0

        max_contacts = 10 * cp_cell_indices.shape[0]
        self.collision_indices_a = wp.empty(max_contacts, dtype=int)
        self.collision_indices_b = wp.empty(max_contacts, dtype=int)
        self.collision_normals = wp.empty(max_contacts, dtype=wp.vec3)
        self.collision_kinematic_gaps = wp.empty(max_contacts, dtype=wp.vec3)

        jac_cols = sim.u_field.space_partition.node_count()
        self._collision_jacobian_a = sp.bsr_zeros(0, jac_cols, block_type=wp.mat33)
        self._collision_jacobian_b = sp.bsr_zeros(0, jac_cols, block_type=wp.mat33)

        self._collision_jacobian = sp.bsr_zeros(0, jac_cols, block_type=wp.mat33)
        self._collision_jacobian_t = sp.bsr_zeros(jac_cols, 0, block_type=wp.mat33)

        self._HtH_work_arrays = sp.bsr_mm_work_arrays()
        self._HbHa_work_arrays = sp.bsr_axpy_work_arrays()

        self._collision_stiffness = self.args.collision_stiffness * self.args.density / n_cp

    def set_collision_quadrature(self, quadrature: fem.PicQuadrature):
        self.collision_quadrature = quadrature

    def add_collision_energy(self, E: float):
        if self.n_contact == 0:
            return E

        cp_du = self._sample_cp_displacement(self.sim.du_field)
        col_energies = wp.empty(self.n_contact, dtype=float)
        wp.launch(
            collision_energy,
            dim=self.n_contact,
            inputs=[
                self.args.collision_radius,
                self.args.friction,
                self.args.dt * self.args.friction_reg,
                self.args.friction_fluid * self.args.friction_reg,
                cp_du,
                self.collision_kinematic_gaps,
                self.collision_normals,
                self.collision_indices_a,
                self.collision_indices_b,
                col_energies,
            ],
        )
        return E + self._collision_stiffness * wp.utils.array_sum(col_energies)

    def add_collision_hessian(self, lhs: wp.array):
        # contacts
        if self.n_contact == 0:
            return lhs

        H = self._collision_jacobian
        Ht = self._collision_jacobian_t
        wp.launch(
            bsr_mul_diag,
            dim=(Ht.nnz, Ht.block_shape[0]),
            inputs=[Ht.scalar_values, Ht.columns, self._col_energy_hessian],
        )

        sp.bsr_mm(
            x=Ht,
            y=H,
            z=lhs,
            alpha=self._collision_stiffness,
            beta=1.0,
            work_arrays=self._HtH_work_arrays,
        )

        return lhs

    def add_collision_forces(self, rhs: wp.array):
        if self.n_contact == 0:
            return rhs

        # contacts
        sp.bsr_mv(
            A=self._collision_jacobian_t,
            x=self._col_energy_gradients,
            y=rhs,
            alpha=-self._collision_stiffness,
            beta=1.0,
        )

        return rhs

    def prepare_newton_step(self, dt: float):
        self.detect_collisions(dt)
        self.build_collision_jacobian()

        # compute per-contact forces and hessian
        n_contact = self.n_contact
        if n_contact > 0:
            self._col_energy_gradients = wp.empty(n_contact, dtype=wp.vec3)
            self._col_energy_hessian = wp.empty(n_contact, dtype=wp.mat33)
            cp_du = self._sample_cp_displacement(self.sim.du_field)

            wp.launch(
                collision_gradient_and_hessian,
                dim=n_contact,
                inputs=[
                    self.args.collision_radius,
                    self.args.friction,
                    dt * self.args.friction_reg,
                    self.args.friction_fluid * self.args.friction_reg,
                    cp_du,
                    self.collision_kinematic_gaps,
                    self.collision_normals,
                    self.collision_indices_a,
                    self.collision_indices_b,
                    self._col_energy_gradients,
                    self._col_energy_hessian,
                ],
            )

    def cp_world_position(self, dest=None):
        cp_pic = self.collision_quadrature
        if dest is None:
            dest = wp.empty(cp_pic.total_point_count(), dtype=wp.vec3)
        fem.interpolate(
            world_position,
            fields={"u": self.sim.u_field},
            dest=dest,
            quadrature=cp_pic,
        )

        return dest

    def _sample_cp_displacement(self, du_field, dest=None):
        cp_pic = self.collision_quadrature
        if dest is None:
            dest = wp.empty(cp_pic.total_point_count(), dtype=wp.vec3)
        fem.interpolate(
            du_field,
            dest=dest,
            quadrature=cp_pic,
        )
        return dest

    def detect_collisions(self, dt):
        max_contacts = self.collision_normals.shape[0]

        count = wp.zeros(1, dtype=int)
        indices_a = self.collision_indices_a
        indices_b = self.collision_indices_b
        normals = self.collision_normals
        kinematic_gaps = self.collision_kinematic_gaps

        self.run_collision_detectors(
            dt,
            count,
            indices_a,
            indices_b,
            normals,
            kinematic_gaps,
        )

        self.n_contact = int(count.numpy()[0])

        if self.n_contact > max_contacts:
            print("Warning: contact buffer size exceeded, some have bee ignored")
            self.n_contact = max_contacts

    def run_collision_detectors(
        self,
        dt,
        count,
        indices_a,
        indices_b,
        normals,
        kinematic_gaps,
    ):
        cp_pic = self.collision_quadrature
        n_cp = cp_pic.total_point_count()
        max_contacts = self.collision_normals.shape[0]

        cp_cur_pos = self.cp_world_position()
        cp_du = self._sample_cp_displacement(self.sim.du_field)

        collision_radius = self.args.collision_radius * self.args.collision_detection_ratio

        if self.args.ground:
            ground_height = self.args.ground_height
            wp.launch(
                detect_ground_collisions,
                dim=n_cp,
                inputs=[
                    max_contacts,
                    self.args.up_axis,
                    cp_cur_pos,
                    cp_du,
                    collision_radius,
                    ground_height,
                    count,
                    normals,
                    kinematic_gaps,
                    indices_a,
                    indices_b,
                ],
            )
        if self.warp_meshes:
            mesh_ids = wp.array([mesh.id for mesh in self.warp_meshes], dtype=wp.uint64)
            wp.launch(
                detect_mesh_collisions,
                dim=(len(mesh_ids), n_cp),
                inputs=[
                    max_contacts,
                    dt,
                    mesh_ids,
                    cp_cur_pos,
                    cp_du,
                    collision_radius,
                    count,
                    normals,
                    kinematic_gaps,
                    indices_a,
                    indices_b,
                ],
            )

    def build_collision_jacobian(self):
        n_contact = self.n_contact

        # Build collision jacobian
        # (derivative of collision gap `pos_a - pos_b` w.r.t. degrees of freedom)

        if n_contact == 0:
            return

        a_cells = wp.empty(n_contact, dtype=int)
        a_coords = wp.empty(n_contact, dtype=wp.vec3)
        b_cells = wp.empty(n_contact, dtype=int)
        b_coords = wp.empty(n_contact, dtype=wp.vec3)
        wp.launch(
            gather_cell_coordinates,
            dim=n_contact,
            inputs=[
                self.collision_quadrature.cell_indices,
                self.collision_quadrature.particle_coords,
                self.collision_indices_a,
                a_cells,
                a_coords,
            ],
        )
        wp.launch(
            gather_cell_coordinates,
            dim=n_contact,
            inputs=[
                self.collision_quadrature.cell_indices,
                self.collision_quadrature.particle_coords,
                self.collision_indices_b,
                b_cells,
                b_coords,
            ],
        )

        measures = wp.ones(n_contact, dtype=float)

        a_contact_pic = fem.PicQuadrature(
            self.collision_quadrature.domain,
            positions=(a_cells, a_coords),
            measures=measures,
        )
        b_contact_pic = fem.PicQuadrature(
            self.collision_quadrature.domain,
            positions=(b_cells, b_coords),
            measures=measures,
        )
        u_trial = fem.make_trial(self.sim.u_field.space, space_partition=self.sim.u_field.space_partition)

        sp.bsr_set_zero(
            self._collision_jacobian_a,
            n_contact,
            self.sim.u_field.space_partition.node_count(),
        )
        fem.interpolate(
            u_trial,
            quadrature=a_contact_pic,
            dest=self._collision_jacobian_a,
            bsr_options={"prune_numerical_zeros": False},
        )

        sp.bsr_set_zero(
            self._collision_jacobian_b,
            n_contact,
            self.sim.u_field.space_partition.node_count(),
        )
        fem.interpolate(
            u_trial,
            quadrature=b_contact_pic,
            dest=self._collision_jacobian_b,
            bsr_options={"prune_numerical_zeros": False},
        )

        self._collision_jacobian_a.nnz_sync()
        self._collision_jacobian_b.nnz_sync()

        sp.bsr_assign(self._collision_jacobian, src=self._collision_jacobian_a)
        sp.bsr_axpy(
            x=self._collision_jacobian_b,
            y=self._collision_jacobian,
            alpha=-1,
            beta=1,
            work_arrays=self._HbHa_work_arrays,
        )

        sp.bsr_set_transpose(dest=self._collision_jacobian_t, src=self._collision_jacobian)


@wp.kernel
def bsr_mul_diag(
    Bt_values: wp.array3d(dtype=float),
    Bt_columns: wp.array(dtype=int),
    C_values: wp.array(dtype=Any),
):
    i, r = wp.tid()
    col = Bt_columns[i]

    C = C_values[col]

    Btr = Bt_values[i, r]
    BtC = wp.vec3(Btr[0], Btr[1], Btr[2]) @ C
    for k in range(3):
        Btr[k] = BtC[k]


@wp.kernel
def detect_ground_collisions(
    max_contacts: int,
    up_axis: int,
    pos_cur: wp.array(dtype=wp.vec3),
    du_cur: wp.array(dtype=wp.vec3),
    radius: float,
    ground_height: float,
    count: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    i = wp.tid()
    x = pos_cur[i]

    if x[up_axis] < ground_height + radius:
        idx = wp.atomic_add(count, 0, 1)
        if idx >= max_contacts:
            return

        nor = fem.linalg.basis_element(wp.vec3(), up_axis)

        normals[idx] = nor
        kinematic_gaps[idx] = (wp.dot(x - du_cur[i], nor) - ground_height) * nor
        indices_a[idx] = i
        indices_b[idx] = fem.NULL_QP_INDEX


@wp.kernel
def detect_mesh_collisions(
    max_contacts: int,
    dt: float,
    mesh_ids: wp.array(dtype=wp.uint64),
    pos_cur: wp.array(dtype=wp.vec3),
    du_cur: wp.array(dtype=wp.vec3),
    radius: float,
    count: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    m, tid = wp.tid()
    mesh_id = mesh_ids[m]

    x = pos_cur[tid]

    query = wp.mesh_query_point(mesh_id, x, radius)

    if query.result:
        cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)

        delta = x - cp
        dist = wp.length(delta) * query.sign

        if dist < radius:
            idx = wp.atomic_add(count, 0, 1)
            if idx >= max_contacts:
                return

            if dist < 0.00001:
                n = wp.mesh_eval_face_normal(mesh_id, query.face)
            else:
                n = wp.normalize(delta) * query.sign
            normals[idx] = n

            v = wp.mesh_eval_velocity(mesh_id, query.face, query.u, query.v)

            kinematic_gap = (dist - wp.dot(du_cur[tid], n)) * n - v * dt
            kinematic_gaps[idx] = kinematic_gap
            indices_a[idx] = tid
            indices_b[idx] = fem.NULL_QP_INDEX


@wp.func
def collision_offset(
    c: int,
    du_cur: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    idx_a = indices_a[c]
    idx_b = indices_b[c]

    offset = du_cur[idx_a] + kinematic_gaps[c]
    if idx_b != fem.NULL_QP_INDEX:
        offset -= du_cur[idx_b]

    return offset


@wp.func
def collision_target_distance(
    c: int,
    radius: float,
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    return wp.select(indices_b[c] == fem.NULL_ELEMENT_INDEX, 2.0, 1.0) * radius


@wp.kernel
def collision_energy(
    radius: float,
    mu: float,
    dt: float,
    nu: float,
    du_cur: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    energies: wp.array(dtype=float),
):
    c = wp.tid()

    offset = collision_offset(c, du_cur, kinematic_gaps, indices_a, indices_b)
    rc = collision_target_distance(c, radius, indices_a, indices_b)

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    stick = wp.select(d_hat < 1.0, 0.0, 1.0)
    gap = d_hat - 1.0
    E = 0.5 * stick * gap * gap

    vt = (offset - d * nor) / dt  # tangential velocity
    vt_norm = wp.length(vt)

    mu_fn = -mu * wp.min(0.0, gap) / rc  # yield force

    E += (
        mu_fn
        * dt
        * (
            0.5 * nu * vt_norm * vt_norm
            + wp.select(
                vt_norm < 1.0,
                vt_norm - 1.0 / 3.0,
                vt_norm * vt_norm * (1.0 - vt_norm / 3.0),
            )
        )
    )

    energies[c] = E


@wp.kernel
def collision_gradient_and_hessian(
    radius: float,
    mu: float,
    dt: float,
    nu: float,
    du_cur: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    gradient: wp.array(dtype=wp.vec3),
    hessian: wp.array(dtype=wp.mat33),
):
    c = wp.tid()

    offset = collision_offset(c, du_cur, kinematic_gaps, indices_a, indices_b)
    rc = collision_target_distance(c, radius, indices_a, indices_b)

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    stick = wp.select(d_hat < 1.0, 0.0, 1.0)

    dE_d_hat = d_hat - 1.0
    gradient[c] = dE_d_hat * stick / rc * nor
    hessian[c] = wp.outer(nor, nor) * stick / (rc * rc)

    vt = (offset - d * nor) / dt  # tangential velocity
    vt_norm = wp.length(vt)
    vt_dir = wp.normalize(vt)  # avoids dealing with 0

    mu_fn = -mu * wp.min(0.0, dE_d_hat) / rc  # yield force

    f1_over_vt_norm = wp.select(vt_norm < 1.0, 1.0 / vt_norm, 2.0 - vt_norm)
    gradient[c] += mu_fn * (f1_over_vt_norm + nu) * vt

    # regularization such that f / H dt <= k v (penalizes friction switching dir)
    friction_slip_reg = 0.1
    df1_d_vtn = wp.max(
        2.0 * (1.0 - vt_norm),
        friction_slip_reg / (0.5 * friction_slip_reg + vt_norm),
    )

    vt_perp = wp.cross(vt_dir, nor)
    hessian[c] += (
        mu_fn / dt * ((df1_d_vtn + nu) * wp.outer(vt_dir, vt_dir) + (f1_over_vt_norm + nu) * wp.outer(vt_perp, vt_perp))
    )


@wp.kernel
def gather_cell_coordinates(
    qp_cells: wp.array(dtype=int),
    qp_coords: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    cells: wp.array(dtype=int),
    coords: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    qp = indices[i]

    if qp == fem.NULL_QP_INDEX:
        cells[i] = fem.NULL_ELEMENT_INDEX
    else:
        cells[i] = qp_cells[qp]
        coords[i] = qp_coords[qp]


@fem.integrand
def world_position(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    return domain(s) + u(s)
