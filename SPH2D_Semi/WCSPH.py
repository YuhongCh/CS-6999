import taichi as ti
import taichi.math as tm
import numpy as np

from BaseSPH import BaseSPH
from ParticleSystem import ParticleSystem, ParticleType
from NeighborSearch import NeighborSearch
from Config import SPHConfig


@ti.data_oriented
class WCSPH(BaseSPH):
    def __init__(self, config: SPHConfig, particle_system: ParticleSystem, neighbor_search: NeighborSearch):
        super().__init__(config, particle_system, neighbor_search)
        self.EOS_expo = 7.0
        self.EOS_coef = 50000.0

        self.pressure = self.particle_system.pressure

    @ti.kernel
    def compute_pressure(self):
        for pid in range(self.num_particle):
            density = self.particle_system.density[pid]
            pressure = self.EOS_coef * (ti.pow(density / self.particle_system.ref_density, self.EOS_expo) - 1)
            self.pressure[pid] = pressure

    @ti.func
    def compute_pressure_force_task(self, pi: int, pj: int, ret: ti.template()):
        """ Use the symmetric formula for discretization of differential operators """
        x_ij = self.particle_system.position[pi] - self.particle_system.position[pj]
        grad_weight = self.kernel.get_gradient_weight(x_ij, self.particle_system.support_radius)
        mass = self.particle_system.ref_mass
        sym_comp = (self.pressure[pi] / ti.pow(self.particle_system.density[pi], 2) +
                    self.pressure[pj] / ti.pow(self.particle_system.density[pj], 2))
        ret += -mass * sym_comp * grad_weight

    @ti.kernel
    def compute_pressure_force(self):
        """ Use the symmetric formula for discretization of differential operators """
        for pid in range(self.num_particle):
            if self.particle_system.type[pid] == ParticleType.Fluid:
                pressure_dv = tm.vec2(0, 0)
                self.neighbor_search.for_each_neighbors(pid, self.compute_pressure_force_task, pressure_dv)
                self.d_velocity[pid] += pressure_dv

    @ti.kernel
    def apply_force(self):
        for pid in range(self.num_particle):
            if self.particle_system.type[pid] == ParticleType.Fluid:
                self.particle_system.acceleration[pid] = self.d_velocity[pid]

    # @ti.kernel
    # def __reset_pressure(self):
        # for pid in range(self.num_particle):
        #     self.pressure_force[pid] = tm.vec2(0, 0)

    # def reset_derived_parameters(self):
    #     self.__reset_pressure()

    def dump(self):
        pid = 0
        print(
            f"{self.particle_system.dt[None]}\t"
            f"{self.particle_system.position[pid]}\t {self.particle_system.velocity[pid]}\t"
            f"{self.particle_system.density[pid]}\t {self.particle_system.pressure[pid]}\t "
            f"{self.d_velocity[pid]}"
            # f"{self.external_force[pid]}\t{self.viscosity_force[pid]}\t{self.pressure_force[pid]}\t{self.particle_system.acceleration[pid]}"
        )
        assert(not np.isnan(self.d_velocity[pid].x))

    def simulate(self):
        self.compute_density()
        self.compute_external_force()
        self.compute_viscosity_force()
        self.compute_pressure()
        self.compute_pressure_force()
        self.apply_force()
        self.dump()
        self.particle_system.step()
        self.reset_parameters()
        self.particle_system.reset_parameters()
