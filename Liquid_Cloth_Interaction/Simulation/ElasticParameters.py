import taichi as ti
import taichi.math as tm


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
