import taichi as ti
import taichi.math as tm
import math

"""
Kernel formula in 3D setting
"""

@ti.data_oriented
class SmoothKernel:
    """ I have no idea why poly6 kernel has to be used with grad of spiky kernel """
    @ti.func
    def t_GetWeight(self, r: tm.vec3, h: float):
        """ poly6 kernel """
        res = 0.0
        r_norm = tm.length(r)
        if r_norm < h:
            coef = 315.0 / (64.0 * math.pi * ti.pow(h, 9))
            res = coef * ti.pow(h**2 - r_norm**2, 3)
        return res

    @ti.func
    def t_GetGrad1Weight(self, r: tm.vec3, h: float):
        """ grad1 of spiky kernel"""
        res = 0.0
        r_norm = tm.length(r)
        if r_norm < h:
            res = -30.0 / (math.pi * h ** 3) * ti.pow(1.0 - r_norm / h, 2)
        return res

    @ti.func
    def t_GetGrad2Weight(self, r: tm.vec3, h: float):
        """ grad2 of spiky kernel"""
        res = 0.0
        r_norm = tm.length(r)
        if r_norm < h:
            x = 1.0 - r_norm / h
            res = 60.0 / (math.pi * h ** 4) * x
        return res


@ti.data_oriented
class CubicSplineKernel:
    """ TODO: Fix the bug here """
    @ti.func
    def t_GetWeight(self, r: tm.vec3, h: float):
        res = 0.0
        q = tm.length(r) / h
        coef = 8 / (math.pi * h ** 3)
        if q <= 1.0:
            if q <= 0.5:
                res = 6 * (q ** 3 - q ** 2) + 1
            else:
                res = 2 * (1 - q) ** 3
        return coef * res

    @ti.func
    def t_GetGrad1Weight(self, r: tm.vec3, h: float) -> tm.vec3:
        res = tm.vec3(0, 0, 0)
        coef = 8 / (math.pi * h ** 3)
        r_len = tm.length(r)
        q = r_len / h
        grad_q = r / (r_len * h)
        if q <= 1.0:
            if q <= 0.5:
                res = 6 * (3 * q * q - 2 * q) * grad_q
            else:
                res = 6 * ti.pow(1 - q, 2) * -grad_q
        return coef * res


class KernelManager:
    @staticmethod
    def p_GetKernel(kernel_type: int):
        if kernel_type == 1:
            return CubicSplineKernel()
        elif kernel_type == 2:
            return SmoothKernel()
        else:
            raise NotImplementedError
