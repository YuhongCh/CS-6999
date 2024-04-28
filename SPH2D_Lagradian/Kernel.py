import taichi as ti
import taichi.math as tm

import math

KERNEL_POSITION_ERROR = False

@ti.data_oriented
class CubicSplineKernel:
    @ti.func
    def get_weight(self, r: tm.vec2, h: float):
        res = 0.0
        q = r.norm() / h
        coef = 40 / (7 * math.pi * h ** 2)
        if q <= 1.0:
            if q <= 0.5:
                res = 6 * (q ** 3 - q ** 2) + 1
            else:
                res = 2 * ti.pow(1 - q, 3)
        return coef * res

    @ti.func
    def get_gradient_weight(self, r: tm.vec2, h: float) -> tm.vec3:
        r_len = tm.length(r)
        if r_len < 1e-6:
            KERNEL_POSITION_ERROR = True

        res = tm.vec2(0, 0)
        coef = 40 / (7 * math.pi * h ** 2)
        q = r_len / h
        grad_q = r / (r_len * h)
        if q <= 1.0:
            if q <= 0.5:
                res = 6 * (3 * q * q - 2 * q) * grad_q
            else:
                res = 6 * ti.pow(1 - q, 2) * -grad_q
        return coef * res
