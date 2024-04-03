import taichi as ti
import taichi.math as tm

from Liquid_Hair_Interaction.Simulation.SPH.SPHConfig import SPHConfig
from Liquid_Hair_Interaction.Simulation.SPH.Boundary import *


@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SPHConfig):
        # Copy configuration
        self.ref_density = config.ref_density
        self.viscosity = config.viscosity
        self.particle_radius = config.particle_radius
        self.particle_mass = config.particle_mass
        self.support_radius = config.support_radius
        self.mass = config.particle_mass
        self.resti_coef = config.resti_coef

        # setup initial value
        self.num_particle = 0
        self.point_list = []
        self.start_domain = ti.field(dtype=float, shape=3)
        self.end_domain = ti.field(dtype=float, shape=3)
        for i in range(3):
            self.start_domain[i] = 10000
            self.end_domain[i] = -10000

        self.position = ti.Vector.field(3, dtype=float)
        self.velocity = ti.Vector.field(3, dtype=float)
        self.acceleration = ti.Vector.field(3, dtype=float)
        self.density = ti.field(dtype=float)

    def p_add_particle(self, pos: tm.vec3):
        self.num_particle += 1
        self.point_list.append(pos)
        for i in range(3):
            self.start_domain[i] = min(self.start_domain[i], pos[i])
            self.end_domain[i] = max(self.end_domain[i], pos[i])

    def p_setup_data(self):
        ti.root.dense(ti.i, self.num_particle).place(self.acceleration)
        ti.root.dense(ti.i, self.num_particle).place(self.velocity)
        ti.root.dense(ti.i, self.num_particle).place(self.position)
        ti.root.dense(ti.i, self.num_particle).place(self.density)

    @ti.func
    def t_GetNumParticle(self) -> int:
        return self.num_particle[type]

    def p_GetNumParticle(self) -> int:
        return self.num_particle[type]
    
    @ti.func
    def t_GetCFL(self) -> float:
        return 0.4 * self.support_radius / 200
    
    def p_GetCFL(self) -> float:
        return 0.4 * self.support_radius / 200

    @ti.kernel
    def t_AdvanceTimeStep(self, dt: float):
        # first integrate the force
        for part_idx in self.position:
            self.velocity[part_idx] += self.acceleration[part_idx] * dt
            self.position[part_idx] += self.velocity[part_idx] * dt

    @ti.kernel
    def t_Clear(self):
        for part_idx in self.acceleration:
            self.acceleration[part_idx] = tm.vec3(0, 0, 0)

    @ti.kernel
    def t_ComputeBoundaryCollision(self):
        for i in self.position:
            if self.position[i].x < self.start_domain.x:
                self.position[i].x = self.start_domain.x
                self.velocity[i].x *= -self.resti_coef

            elif self.position[i].x >= self.end_domain.x:
                self.position[i].x = self.end_domain.x
                self.velocity[i].x *= -self.resti_coef

            if self.position[i].y < self.start_domain.y:
                self.position[i].y = self.start_domain.y
                self.velocity[i].y *= -self.resti_coef

            elif self.position[i].y >= self.end_domain.y:
                self.position[i].y = self.end_domain.y
                self.velocity[i].y *= -self.resti_coef

            if self.position[i].z < self.start_domain.z:
                self.position[i].z = self.start_domain.z
                self.velocity[i].z *= -self.resti_coef

            elif self.position[i].z >= self.end_domain.z:
                self.position[i].z = self.end_domain.z
                self.velocity[i].z *= -self.resti_coef



    # @ti.kernel
    # def t_Initialize(self, center: tm.vec3):
    #     num_part_per_row = tm.ceil(tm.sqrt(tm.sqrt(self.num_particle)))
    #     interval = self.particle_radius * 2
    #     for part_idx in self.particles:
    #         num_part_per_level = num_part_per_row * num_part_per_row
    #         y_idx = part_idx // num_part_per_level
    #         x_idx = (part_idx - y_idx * num_part_per_level) // num_part_per_row
    #         z_idx = part_idx - y_idx * num_part_per_level - x_idx * num_part_per_row
    #         pos = center + tm.vec3(x_idx, y_idx, z_idx) * interval
    #         self.particles[part_idx].x = pos
    #         self.particles[part_idx].mass = self.particle_mass
    #         self.particles[part_idx].radius = self.particle_radius