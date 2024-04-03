import taichi as ti
import taichi.math as tm


"""
Boundary type is defined as following
1: Sphere   -> radius: float
2: Box      -> radius: float, lengths: tm.vec3
3: Capsule  -> radius: float, half_length: float
4: Union
5: Intersect
6: Count
"""
SPHERE = 1
BOX = 2
CAPSULE = 3

@ti.data_oriented
class BaseBoundary:
    def __init__(self, type: int, center: tm.vec3, params: dict):
        self.type = type
        self.center = center
        self.params = params

    @ti.func
    def t_GetSDF(self, pos: tm.vec3) -> float:
        dist = 0.0
        if type == SPHERE:
            dist = tm.length(pos - self.center) - self.radius
        elif type == BOX:
            lengths = self.params["lengths"]
            dx = abs(pos.x - self.center.x) - lengths.x
            dy = abs(pos.y - self.center.y) - lengths.y
            dz = abs(pos.z - self.center.z) - lengths.z
            pdx = max(dx, 0.0)
            pdy = max(dy, 0.0)
            pdz = max(dz, 0.0)
            dist = min(max(max(dx, dy), dz), 0.0) + tm.sqrt(pdx * pdx + pdy * pdy + pdz * pdz) - self.radius
        else:
            assert 1 == 2
        return dist



    
