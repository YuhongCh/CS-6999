import taichi as ti
import taichi.math as tm




@ti.data_oriented
class Integrator:
    def __init__(self, dt: float, type: str, criterion: float):
        self.dt = dt
        self.type = type
        self.criterion = criterion
