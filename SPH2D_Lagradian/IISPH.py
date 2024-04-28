import taichi as ti
import taichi.math as tm
import numpy as np

from BaseSPH import BaseSPH
from ParticleSystem import ParticleSystem, ParticleType
from NeighborSearch import NeighborSearch
from Config import SPHConfig


@ti.data_oriented
class IISPH(BaseSPH):
    def __init__(self, config: SPHConfig, particle_system: ParticleSystem, neighbor_search: NeighborSearch):
        super().__init__(config, particle_system, neighbor_search)

        self.diag = ti.field(float, shape=self.num_particle)
        self.source = ti.field(float, shape=self.num_particle)
        self.pressure = self.particle_system.pressure

        self.buffer = ti.Vector.field(2, dtype=float, shape=())
        self.pressure_dv = ti.Vector.field(2, dtype=float, shape=self.num_particle)
        self.ap = ti.field(float, shape=self.num_particle)

    @ti.kernel
    def compute_advect_velocity(self):
        for pid in range(self.num_particle):
            if self.particle_system.type[pid] == ParticleType.Fluid:
                delta_velocity = self.d_velocity[pid] * self.particle_system.dt[None]
                self.particle_system.velocity[pid] += delta_velocity

    @ti.func
    def compute_source_task(self, pi: int, pj: int, ret: ti.template()):
        x_ij = self.particle_system.position[pi] - self.particle_system.position[pj]
        advect_v_ij = self.particle_system.velocity[pi] - self.particle_system.velocity[pj]
        grad_weight = self.kernel.get_gradient_weight(x_ij, self.particle_system.support_radius)
        ret += self.particle_system.ref_mass * tm.dot(advect_v_ij, grad_weight)

    @ti.func
    def compute_diag_task2(self, pi: int, pj: int, ret: ti.template()):
        x_ij = self.particle_system.position[pi] - self.particle_system.position[pj]
        grad_weight = self.kernel.get_gradient_weight(x_ij, self.particle_system.support_radius)
        ret += self.particle_system.ref_mass / ti.pow(self.particle_system.density[pj], 2) * grad_weight

    @ti.func
    def compute_diag_task1(self, pi: int, pj: int, ret: ti.template()):
        x_ij = self.particle_system.position[pi] - self.particle_system.position[pj]
        coef = self.particle_system.ref_mass / ti.pow(self.particle_system.density[pi], 2)
        grad_weight = self.kernel.get_gradient_weight(x_ij, self.particle_system.support_radius)
        ret += self.particle_system.ref_mass * tm.dot(self.buffer[None] + coef * grad_weight, grad_weight)

    @ti.kernel
    def compute_diag_source(self):
        for pid in range(self.num_particle):

            src_val = 0.0
            self.neighbor_search.for_each_neighbors(pid, self.compute_source_task, src_val)
            self.source[pid] = (self.particle_system.ref_density - self.particle_system.density[pid] -
                                self.particle_system.dt[None] * src_val)

            diag_val = 0.0
            self.buffer[None] = tm.vec2(0, 0)
            self.neighbor_search.for_each_neighbors(pid, self.compute_diag_task2, self.buffer[None])
            self.neighbor_search.for_each_neighbors(pid, self.compute_diag_task1, diag_val)
            self.diag[pid] = -ti.pow(self.particle_system.dt[None], 2) * diag_val

    @ti.func
    def compute_Ap_task(self, pi, pj, ret: ti.template()):
        x_ij = self.particle_system.position[pi] - self.particle_system.position[pj]
        grad_weight = self.kernel.get_gradient_weight(x_ij, self.particle_system.support_radius)
        comp1 = self.particle_system.ref_mass * (self.pressure_dv[pi] - self.pressure_dv[pj])
        ret += tm.dot(comp1, grad_weight)

    @ti.func
    def compute_Ap(self):
        for pid in range(self.num_particle):
            Ap = 0.0
            self.neighbor_search.for_each_neighbors(pid, self.compute_Ap_task, Ap)
            self.ap[pid] = ti.pow(self.particle_system.dt[None], 2) * Ap

    @ti.func
    def compute_error(self) -> float:
        error = 0.0
        for pid in range(self.num_particle):
            error += ti.abs(self.ap[pid] - self.source[pid]) / self.particle_system.ref_density
            if tm.isnan(self.ap[pid]):
                print(f"{pid} has nan ap")
            if tm.isnan(self.source[pid]):
                print(f"{pid} has nan source")
        return error / self.num_particle

    @ti.kernel
    def compute_pressure(self):
        relax = 0.5
        threshold = 0.0001

        # generate a possible guess with pressure function in WCSPH
        for pid in range(self.num_particle):
            density = self.particle_system.density[pid]
            pressure = 5000.0 * (ti.pow(density / self.particle_system.ref_density,7) - 1)
            self.pressure[pid] = pressure
        # self.pressure.fill(0.0)

        # actually compute the pressure
        self.compute_pressure_dv()
        self.compute_Ap()
        error = self.compute_error()
        iter_count = 0
        while error > threshold:
            for pid in range(self.num_particle):
                pressure = self.pressure[pid] + relax / self.diag[pid] * (self.source[pid] - self.ap[pid])
                self.pressure[pid] = ti.max(pressure, 0.0)
            self.compute_pressure_dv()
            self.compute_Ap()
            error = self.compute_error()
            iter_count += 1
            print(f"iteration {iter_count} has error {error}")

    @ti.func
    def compute_pressure_force_task(self, pi: int, pj: int, ret: ti.template()):
        """ Use the symmetric formula for discretization of differential operators """
        x_ij = self.particle_system.position[pi] - self.particle_system.position[pj]
        grad_weight = self.kernel.get_gradient_weight(x_ij, self.particle_system.support_radius)
        mass = self.particle_system.ref_mass
        sym_comp = (self.pressure[pi] / ti.pow(self.particle_system.density[pi], 2) +
                    self.pressure[pj] / ti.pow(self.particle_system.density[pj], 2))
        ret += -mass * sym_comp * grad_weight

    @ti.func
    def compute_pressure_dv(self):
        """ Use the symmetric formula for discretization of differential operators """
        for pid in range(self.num_particle):
            self.pressure_dv[pid] = tm.vec2(0, 0)
            if self.particle_system.type[pid] == ParticleType.Fluid:
                self.neighbor_search.for_each_neighbors(pid, self.compute_pressure_force_task, self.pressure_dv[pid])

    @ti.kernel
    def apply_pressure_dv(self):
        for pid in range(self.num_particle):
            if self.particle_system.type[pid] == ParticleType.Fluid:
                self.particle_system.velocity[pid] += self.pressure_dv[pid] * self.particle_system.dt[None]

    def dump(self):
        pid = 0
        print(
            f"{self.particle_system.dt[None]}\t"
            f"{self.particle_system.position[pid]}\t {self.particle_system.velocity[pid]}\t"
            f"{self.particle_system.density[pid]}\t {self.particle_system.pressure[pid]}\t "
            f"{self.d_velocity[pid]}\t {self.pressure_dv[pid]}\t"
            f"{self.ap[pid]}\t {self.source[pid]}"
        )
        assert(not np.isnan(self.d_velocity[pid].x))

    def simulate(self):
        self.compute_density()
        self.compute_external_force()
        self.compute_viscosity_force()
        self.compute_advect_velocity()
        self.compute_diag_source()

        self.compute_pressure()
        self.apply_pressure_dv()

        # self.dump()
        self.particle_system.step()

        # self.particle_system.compute_CFL()

        self.reset_parameters()
        self.particle_system.reset_parameters()
