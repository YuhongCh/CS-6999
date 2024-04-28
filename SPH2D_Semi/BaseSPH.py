import taichi as ti
import taichi.math as tm

import math

from ParticleSystem import ParticleSystem, ParticleType
from Config import SPHConfig
from NeighborSearch import NeighborSearch
from Kernel import CubicSplineKernel as Kernel
from Boundary import Boundary


@ti.data_oriented
class BaseSPH:
    def __init__(self, config: SPHConfig, particle_system: ParticleSystem, neighbor_search: NeighborSearch):
        self.viscosity_mu = config.viscosity_mu
        self.dt = config.dt
        self.particle_system = particle_system
        self.neighbor_search = neighbor_search
        self.kernel = Kernel()
        self.boundary = Boundary(tm.vec2(0.5, 0.5), tm.vec2(0.5, 0.5), 0.05, config)

        # shallow copy needed variables from particle system
        self.num_particle = self.particle_system.get_particle_num(ParticleType.NonEmpty)

        self.d_velocity = ti.Vector.field(2, dtype=float, shape=self.num_particle)

    @ti.func
    def compute_density_task(self, pi: int, pj: int, ret: ti.template()):
        x_ij = self.particle_system.position[pi] - self.particle_system.position[pj]
        weight = self.kernel.get_weight(x_ij, self.particle_system.support_radius)
        ret += self.particle_system.ref_mass * weight

    @ti.kernel
    def compute_density(self):
        for pid in range(self.num_particle):
            if self.particle_system.type[pid] == ParticleType.Fluid:
                density = 0.0
                self.neighbor_search.for_each_neighbors(pid, self.compute_density_task, density)
                density += self.boundary.get_boundary_density(self.particle_system.position[pid])
                density = ti.max(density, self.particle_system.ref_density)
                self.particle_system.density[pid] = density

    @ti.func
    def compute_viscosity_force_task(self, pi: int, pj: int, ret: ti.template()):
        if self.particle_system.type[pj] == ParticleType.Fluid:
            coef = self.particle_system.ref_mass / self.particle_system.density[pj]
            v_ij = self.particle_system.velocity[pi] - self.particle_system.velocity[pj]
            x_ij = self.particle_system.position[pi] - self.particle_system.position[pj]

            if x_ij.norm() < 1e-8:
                print(pi, pj, self.particle_system.position[pi],
                      self.particle_system.type[pi], self.particle_system.type[pj])
                assert(False)

            h = self.particle_system.support_radius
            grad_weight = self.kernel.get_gradient_weight(x_ij, h)

            ret += coef * tm.dot(v_ij, x_ij) / (tm.dot(x_ij, x_ij) + 0.01 * h * h) * grad_weight

    @ti.kernel
    def compute_viscosity_force(self):
        # TODO: checkout https://peridyno.com/zh/topics/particlesystem/artificalviscosity/ for artificial viscosity
        for pid in range(self.num_particle):
            if self.particle_system.type[pid] == ParticleType.Fluid:
                viscosity = tm.vec2(0, 0)
                self.neighbor_search.for_each_neighbors(pid, self.compute_viscosity_force_task, viscosity)
                viscosity *= self.viscosity_mu * 8 # 8 stands for 2 * (2 + dim)
                self.d_velocity[pid] += viscosity

    @ti.kernel
    def compute_external_force(self):
        for pid in range(self.num_particle):
            # external gravity force
            if self.particle_system.type[pid] == ParticleType.Fluid:
                self.d_velocity[pid] += tm.vec2(0, -9.81)

    @ti.kernel
    def __reset_parameters(self):
        for pid in range(self.num_particle):
            self.d_velocity[pid] = tm.vec2(0, 0)

    def reset_derived_parameters(self):
        pass

    def reset_parameters(self):
        self.__reset_parameters()
        self.reset_derived_parameters()