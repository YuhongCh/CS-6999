import taichi as ti
import taichi.math as tm

from Liquid_Hair_Interaction.Simulation.SPH.SPHConfig import SPHConfig
from Liquid_Hair_Interaction.Simulation.SPH.ParticleSystem import ParticleSystem
from Liquid_Hair_Interaction.Simulation.SPH.BaseSPH import BaseSPH

@ti.data_oriented
class WCSPH(BaseSPH):
    def __init__(self, config: SPHConfig, ps: ParticleSystem):
        super().__init__(config, ps)
        self.EOS_expo = 7
        self.EOS_coef = 50

    @ti.func
    def t_ComputePressureForceTask(self, pi: int, pj: int, ret: ti.template()):
        mass = self.ps.particles[pi].mass

        pi_pos = self.ps.particles[pi].x
        pi_density = self.ps.particles[pi].density
        pi_pressure = self.ps.particles[pi].density
        pj_pos = self.ps.particles[pj].x
        pj_density = self.ps.particles[pj].density
        pj_pressure = self.ps.particles[pj].density

        r = pi_pos - pj_pos
        r_dir = r.normalized()
        grad_weight = self.kernel.t_GetGrad1Weight(r, self.ps.support_radius) * r_dir

        force = mass * mass * (pi_pressure / tm.pow(pi_density, 2) + pj_pressure / tm.pow(pj_density, 2)) * grad_weight
        ret -= force

    @ti.kernel
    def t_ComputePressure(self):
        for part_idx in self.ps.particles:
            density = ti.max(self.ps.particles[part_idx].density, self.ps.ref_density)
            pressure = self.EOS_coef * (ti.pow(density / self.ps.ref_density, self.EOS_expo) - 1.0)
            self.ps.particles[part_idx].pressure = pressure

    @ti.kernel
    def t_ComputePressureForce(self):
        for part_idx in self.ps.particles:
            self.neighbor_search.t_ForEachNeighbors(part_idx,
                                                    self.t_ComputePressureForceTask,
                                                    self.ps.pressure_force[part_idx])

    def p_Update(self):
        super().p_Update()
        self.t_ComputePressureForce()



        