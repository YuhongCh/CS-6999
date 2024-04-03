import taichi as ti
import taichi.math as tm

"""
This files stores all needed initial configuration of SPH model
"""

@ti.dataclass
class SPHConfig:
    # particle configuration
    particle_radius: float
    particle_mass: float
    support_radius: float
    viscosity: float
    stiffness: float
    ref_density: float
    resti_coef: float

    # scene configuration
    grid_radius: float
    max_neighbor_count: int
    max_part_in_grid: int

    # user choice methods
    kernel_type: int
    neighbor_search_type: int
    sph_type: int


    def p_DefaultInit(self):
        # particle configuration
        self.particle_radius = 0.01
        self.particle_mass = 1.0
        self.support_radius = 2 * self.particle_radius
        self.viscosity = 0.01
        self.stiffness = 1000
        self.ref_density = 1000
        self.resti_coef = 0.9

        # neighbor configuration
        self.grid_radius = 0.1
        self.max_neighbor_count = 100
        self.max_part_in_grid = 100

        # SPH and kernel configuration
        self.kernel_type = 1
        self.sph_type = 1



