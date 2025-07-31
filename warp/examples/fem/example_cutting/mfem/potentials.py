# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

import warp as wp
import warp.fem as fem
from warp.fem import Domain, Field, Sample

from .softbody_sim import DisplacementPotential, SoftbodySim


@wp.struct
class VolumetricForces:
    count: int
    centers: wp.array(dtype=wp.vec3)
    radii: wp.array(dtype=float)
    forces: wp.array(dtype=wp.vec3)
    tot_weight: wp.array(dtype=float)


@wp.func
def force_weight(x: wp.vec3, forces: VolumetricForces, force_index: int):
    r = wp.min(
        wp.length(x - forces.centers[force_index]) / (forces.radii[force_index] + 1.0e-7),
        1.0,
    )
    r2 = r * r
    return 2.0 * r * r2 - 3.0 * r2 + 1.0  # cubic spline


@fem.integrand
def force_weight_form(s: Sample, domain: Domain, forces: VolumetricForces, force_index: int):
    return force_weight(domain(s), forces, force_index)


@fem.integrand
def force_action(x: wp.vec3, forces: VolumetricForces, force_index: int, vec: wp.vec3):
    # action of a force over a vector
    return wp.where(
        forces.tot_weight[force_index] >= 1.0e-6,
        wp.dot(forces.forces[force_index], vec) * force_weight(x, forces, force_index) / forces.tot_weight[force_index],
        0.0,
    )


@fem.integrand
def external_forces_form(s: Sample, domain: Domain, v: Field, forces: VolumetricForces):
    f = float(0.0)
    x = domain(s)
    for fi in range(forces.count):
        f += force_action(x, forces, fi, v(s))
    return f


@fem.integrand
def external_forces_potential_energy(
    s: Sample,
    domain: Domain,
    u: Field,
    forces: VolumetricForces,
):
    return -external_forces_form(s, domain, u, forces)


class VolumetricForcePotential(DisplacementPotential):
    def __init__(self, sim, reserve_count: int = 0):
        super().__init__(sim)

        self.forces = VolumetricForces()
        self.forces.count = 0
        self.reserve(reserve_count)

    def init_constant_forms(self):
        self.update_force_weight()

    def reserve(self, count: int):
        self.forces.forces = wp.empty(shape=(count,), dtype=wp.vec3)
        self.forces.radii = wp.empty(shape=(count,), dtype=float)
        self.forces.centers = wp.empty(shape=(count,), dtype=wp.vec3)
        self.forces.tot_weight = wp.empty(shape=(count,), dtype=float)

    def update_force_weight(self):
        for fi in range(self.forces.count):
            wi = self.forces.tot_weight[fi : fi + 1]
            fem.integrate(
                force_weight_form,
                quadrature=self.sim.vel_quadrature,
                values={
                    "force_index": fi,
                    "forces": self.forces,
                },
                output=wi,
                accumulate_dtype=wp.float32,
            )

    def add_forces(self, rhs, _tape):
        if self.forces.count > 0:
            # NOT differentiating with respect to external forces
            # Those are assumed to not depend on the geometry
            fem.integrate(
                external_forces_form,
                fields={"v": self.sim.u_test},
                values={
                    "forces": self.forces,
                },
                output_dtype=wp.vec3,
                quadrature=self.sim.vel_quadrature,
                kernel_options={"enable_backward": False},
                output=rhs,
                add=True,
            )

    def add_energy(self, E_u):
        fem.integrate(
            external_forces_potential_energy,
            quadrature=self.sim.vel_quadrature,
            fields={"u": self.sim.du_field},
            values={
                "forces": self.forces,
            },
            output=E_u,
            add=True,
        )


class CustomPotential(DisplacementPotential):
    def __init__(self, sim, energy_form, forces_form, hessian_form):
        super().__init__(sim)

        self.energy_form = energy_form
        self.forces_form = forces_form
        self.hessian_form = hessian_form

    def add_energy(self, E_u):
        if self.energy_form is not None:
            fem.integrate(
                self.energy_form,
                fields={"u_cur": self.u_field},
                output_dtype=float,
                quadrature=self.vel_quadrature,
                output=E_u,
                add=True,
            )

    def add_forces(self, rhs, tape):
        if self.forces_form is not None:
            with tape:
                fem.integrate(
                    self.forces_form,
                    fields={"u_cur": self.u_field, "v": self.u_test},
                    output=rhs,
                    add=True,
                    quadrature=self.vel_quadrature,
                    kernel_options={"enable_backward": True},
                )

    def add_hessian(self, lhs):
        if self.hessian_form is not None:
            fem.integrate(
                self.hessian_form,
                fields={"u_cur": self.u_field, "u": self.u_trial, "v": self.u_test},
                output_dtype=float,
                quadrature=self.vel_quadrature,
                output=lhs,
                add=True,
            )


class PrescribedMotion(DisplacementPotential):
    def __init__(self, sim: SoftbodySim, quadrature: Optional[fem.PicQuadrature] = None):
        super().__init__(sim)

        self.set_quadrature(quadrature)
        self._prescribed_pos_field = None
        self._prescribed_pos_weight_field = None

    def set_quadrature(self, quadrature: fem.PicQuadrature):
        self.quadrature = quadrature

    def set_prescribed_positions(self, pos_field: fem.field.GeometryField, weight_field: fem.field.GeometryField):
        # for driving objects kinematically
        self._prescribed_pos_field = pos_field
        self._prescribed_pos_weight_field = weight_field

    def init_constant_forms(self):
        if self.quadrature is None:
            self.set_quadrature(self.sim.vel_quadrature)

    def add_energy(self, E: wp.array):
        if self._prescribed_pos_field:
            fem.integrate(
                prescribed_position_energy_form,
                quadrature=self.quadrature,
                fields={
                    "u_cur": self.sim.u_field,
                    "stiffness": self._prescribed_pos_weight_field,
                    "target": self._prescribed_pos_field,
                },
                output=E,
                add=True,
            )

    def add_hessian(self, lhs: wp.array):
        if self._prescribed_pos_field:
            fem.integrate(
                prescribed_position_lhs_form,
                quadrature=self.quadrature,
                fields={
                    "u": self.sim.u_trial,
                    "v": self.sim.u_test,
                    "stiffness": self._prescribed_pos_weight_field,
                },
                output=lhs,
                add=True,
            )

        return lhs

    def add_forces(self, rhs: wp.array, _tape=None):
        if self._prescribed_pos_field:
            fem.integrate(
                prescribed_position_rhs_form,
                quadrature=self.quadrature,
                fields={
                    "u_cur": self.sim.u_field,
                    "v": self.sim.u_test,
                    "stiffness": self._prescribed_pos_weight_field,
                    "target": self._prescribed_pos_field,
                },
                output=rhs,
                add=True,
            )

        return rhs


@fem.integrand
def prescribed_position_lhs_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
    stiffness: fem.Field,
):
    u_displ = u(s)
    v_displ = v(s)

    return stiffness(s) * wp.dot(u_displ, v_displ)


@fem.integrand
def prescribed_position_rhs_form(
    s: fem.Sample,
    domain: fem.Domain,
    u_cur: fem.Field,
    v: fem.Field,
    stiffness: fem.Field,
    target: fem.Field,
):
    pos = u_cur(s) + domain(s)
    v_displ = v(s)
    target_pos = target(s)
    return stiffness(s) * wp.dot(target_pos - pos, v_displ)


@fem.integrand
def prescribed_position_energy_form(
    s: fem.Sample,
    domain: fem.Domain,
    u_cur: fem.Field,
    stiffness: fem.Field,
    target: fem.Field,
):
    pos = u_cur(s) + domain(s)
    target_pos = target(s)
    return 0.5 * stiffness(s) * wp.length_sq(pos - target_pos)
