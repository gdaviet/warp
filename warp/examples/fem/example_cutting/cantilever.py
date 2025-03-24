import argparse

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
from warp.examples.fem.example_cutting.softbody_sim import ClassicFEM

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
    clamped = wp.select(pos[0] < 1.0, 1.0, 0.0)

    return wp.dot(u(s), v(s)) * clamped


def run_softbody_sim(
    sim: ClassicFEM,
    init_callback=None,
    frame_callback=None,
):
    import polyscope as ps

    ps.init()
    ps.set_ground_plane_mode(mode_str="none")

    node_pos = sim.u_field.space.node_positions().numpy()

    tets = sim.u_field.space.node_tets()
    ps_vol = ps.register_volume_mesh("volume mesh", node_pos, tets=tets, edge_width=1.0)

    sim.init_constant_forms()
    sim.project_constant_forms()
    sim.cur_frame = 0

    def callback():
        sim.cur_frame = sim.cur_frame + 1

        with wp.ScopedTimer(f"--- Frame --- {sim.cur_frame}", synchronize=True):
            sim.run_frame()

        displaced_pos = sim.u_field.space.node_positions().numpy()
        displaced_pos += sim.u_field.dof_values.numpy()
        ps_vol.update_vertex_positions(displaced_pos)

        # ps.screenshot()

    ps.set_user_callback(callback)
    ps.show()


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
