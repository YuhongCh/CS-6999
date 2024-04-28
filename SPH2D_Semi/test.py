import taichi as ti
import taichi.math as tm

import math

from Kernel import CubicSplineKernel

ti.init(arch=ti.cpu)

kernel = CubicSplineKernel()

def cubic_kernel(r_norm):
    res = 0.0
    h = 0.04
    # value of cubic spline smoothing kernel
    k = 40 / 7 / math.pi
    k /= h ** 2
    q = r_norm / h
    if q <= 1.0:
        if q <= 0.5:
            q2 = q * q
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1)
        else:
            res = k * 2 * ti.pow(1 - q, 3.0)
    return res


@ti.kernel
def something(x: tm.vec2, r: float) -> float:
    return kernel.get_weight(x, r)


x = tm.vec2(0.01, 0.01)
r = 0.01
mass = 1000.0 * math.pi * r**2
w = something(x, 4 * r)

print(mass, w, mass * w, mass * cubic_kernel(x.norm()))
