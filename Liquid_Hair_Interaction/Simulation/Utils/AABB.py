import taichi as ti
import taichi.math as tm

@ti.data_oriented
class AABB:
    """ Axis-Aligned Bounding Box in 3D """
    def __init__(self, min_corner: tm.vec3, max_corner: tm.vec3):
        self.min_corner = min_corner
        self.max_corner = max_corner