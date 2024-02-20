import taichi as ti
import taichi.math as tm

from Liquid_Cloth_Interaction.Simulation.DER.Strand import DER_Strand

@ti.data_oriented
class Integrator:
    def __init__(self, dt: float, type: str, criterion: float):
        self.dt = dt
        self.type = type
        self.criterion = criterion



