import taichi as ti
import taichi.math as tm

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

        self.pressure = ti.field(float, shape=self.num_particle)
        self.pressure_acc = ti.Vector.field(3, dtype=float, shape=self.num_particle)

    @ti.func
    def get_state_equation(self, density: float) -> float:
        return self.EOS_coef * (ti.pow(density / self.particle_system.ref_density, self.EOS_expo) - 1)

    @ti.kernel
    def compute_pressure(self):
        for pid in range(self.num_particle):
            self.pressure[pid] = self.get_state_equation(self.particle_system.density[pid])

    @ti.func
    def compute_pressure_force_task(self, pi: int, pj: int, ret: ti.template()):
        """ Use the symmetric formula for discretization of differential operators """
        x_ij = self.particle_system.position[pi] - self.particle_system.position[pj]
        grad_weight = self.kernel.get_gradient_weight(x_ij, self.particle_system.support_radius)
        sym_comp = 0.0

        if self.particle_system.type[pj] == ParticleType.Fluid:
            sym_comp = (self.pressure[pi] / ti.pow(self.particle_system.density[pi], 2) +
                        self.pressure[pj] / ti.pow(self.particle_system.density[pj], 2))
        elif self.particle_system.type[pj] == ParticleType.Boundary:
            sym_comp = (self.pressure[pi] / ti.pow(self.particle_system.density[pi], 2) +
                        self.pressure[pj] / ti.pow(self.particle_system.ref_density, 2))

        ret += -self.particle_system.particle_mass * sym_comp * grad_weight

    @ti.kernel
    def compute_pressure_force(self):
        """ Use the symmetric formula for discretization of differential operators """
        for pid in range(self.num_particle):
            self.neighbor_search.for_each_neighbors(pid, self.compute_pressure_force_task, self.pressure_acc[pid])

    @ti.kernel
    def advect(self):
        for pid in range(self.num_particle):
            if self.particle_system.type[pid] == ParticleType.Fluid:
                density = self.particle_system.density[pid]
                self.particle_system.acceleration[pid] += self.external_force[pid] / density
                self.particle_system.acceleration[pid] += self.viscosity_force[pid] / density
                self.particle_system.acceleration[pid] += self.pressure_acc[pid]

                self.particle_system.velocity[pid] += self.particle_system.acceleration[pid] * self.dt
                self.particle_system.position[pid] += self.particle_system.velocity[pid] * self.dt

    @ti.kernel
    def __reset_pressure(self):
        for pid in range(self.num_particle):
            self.pressure_acc[pid] = tm.vec3(0, 0, 0)

    def reset_derived_parameters(self):
        self.__reset_pressure()

    def simulate(self):
        self.compute_density()
        print("complete density computation")
        self._compute_external_force()
        print("complete compute_external_force")
        self._compute_viscosity_force()
        print("complete compute_viscosity_force")
        self.compute_pressure()
        print("complete compute_pressure")
        self.compute_pressure_force()
        print("complete compute_pressure_force")
        self.advect()
        self.reset_parameters()
        self.particle_system.reset_parameters()
