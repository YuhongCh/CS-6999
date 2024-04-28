import taichi as ti
import taichi.math as tm

BOX = 1
CAPSULE = 2

@ti.data_oriented
class SolidBoundary:
    def __init__(self, type: int, center: tm.vec3, params: dict):
        self.type = type
        self.center = center
        self.params = params

    



    
