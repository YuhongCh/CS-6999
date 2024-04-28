import taichi as ti
import taichi.math as tm

import math


@ti.dataclass
class SPHConfig:
    max_particle_count: int
    particle_radius: float  # unit m
    particle_volume: float
    ref_mass: float    # unit kg
    support_radius: float   # unit m
    viscosity_mu: float
    ref_density: float
    c_s: float

    # scene configuration
    dt: float               # unit s
    start_domain: tm.vec2
    end_domain: tm.vec2
    grid_radius: float

    def default_init(self):
        # particle configuration
        self.max_particle_count = 2000
        self.particle_radius = 0.01
        self.support_radius = 4 * self.particle_radius
        self.viscosity_mu = 0.05
        self.ref_density = 1000.0
        self.c_s = 100.0
        self.particle_volume = math.pi * ti.pow(self.particle_radius, 2)
        self.ref_mass = self.ref_density * self.particle_volume

        # neighbor configuration
        self.dt = 0.001
        self.grid_radius = self.support_radius
        self.start_domain = tm.vec2(0, 0)
        self.end_domain = tm.vec2(1, 1)

    def dump(self):
        print(
            f"ref density = {self.ref_density}\t ref volume = {self.particle_volume}\t ref mass = {self.ref_mass}\t"
            f"particle radius = {self.particle_radius}\t support_radius = {self.support_radius}"
        )
