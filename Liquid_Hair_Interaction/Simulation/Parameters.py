"""
This file should only consist of dataclass parameters
Currently includes:
    1) LiquidParameter
    2) ElasticParameter
"""

import taichi as ti
import taichi.math as tm


@ti.dataclass
class LiquidParameter:
    liquid_density: float
    air_density: float
    surface_tension_coef: float
    liquid_viscosity: float
    air_viscosity: float
    yazdchi_power: float
    pore_radius: float
    yarn_diameter: float
    rest_volume_fraction: float
    # lambda:
    cohesion_coef: float
    correction_multiplier: float
    correction_strength: float
    flip_coeff: float
    elasto_flip_asym_coeff: float
    elasto_flip_coeff: float
    elasto_advect_coeff: float
    particle_cell_multiplier: float
    levelset_young_modulus: float
    liquid_boundary_friction: float
    levelset_thickness: float
    elasto_capture_rate: float
    correction_step: int
    bending_scheme: int
    iteration_print_step: int
    surf_tension_smoothing_step: int
    use_surf_tension: bool
    use_cohesion: bool
    solid_cohesion: bool
    soft_cohesion: bool
    solve_solid: bool
    use_nonlinear_drag: bool
    use_drag: bool
    apply_pressure_solid: bool
    use_levelset_force: bool
    apply_pressure_manifold: bool
    use_twist: bool
    use_bicgstab: bool
    use_amgpcg_solid: bool
    use_pcr: bool
    apply_pore_pressure_solid: bool
    propagate_solid_velocity: bool
    check_divergence: bool
    use_varying_fraction: bool
    compute_viscosity: bool
    implicit_viscosity: bool
    drag_by_future_solid: bool
    drag_by_air: bool
    init_nonuniform_fraction: bool
    use_group_precondition: bool
    use_lagrangian_mpm: bool
    use_cosolve_angular: bool


@ti.dataclass
class ElasticParameters:
    radius: float
    youngs_modulus: int
    shear_modulus: int
    poisson_ratios: float
    collision_multiplier: float
    attach_multiplier: float
    density: float
    viscosity: int
    base_rotation: float