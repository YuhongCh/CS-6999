import taichi as ti
import taichi.math as tm

import math


@ti.dataclass
class SPHConfig:
    max_particle_count: int
    particle_radius: float  # unit m
    particle_volume: float
    particle_mass: float    # unit kg
    support_radius: float   # unit m
    viscosity_mu: float
    ref_density: float
    c_s: float

    # scene configuration
    dt: float               # unit s
    start_domain: tm.vec3
    end_domain: tm.vec3
    grid_radius: float

    def default_init(self):
        # particle configuration
        self.max_particle_count = 30000
        self.particle_radius = 0.01
        self.support_radius = 4 * self.particle_radius
        self.viscosity_mu = 0.05
        self.ref_density = 1000.0
        self.c_s = 100.0
        self.particle_volume = (4.0 * math.pi) / 3.0 * ti.pow(self.particle_radius, 3)
        self.particle_mass = self.particle_volume * self.ref_density

        # neighbor configuration
        self.dt = 0.001
        self.grid_radius = 0.02
        self.start_domain = tm.vec3(0, 0, 0)
        self.end_domain = tm.vec3(1, 1, 1)
