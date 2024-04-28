import taichi as ti
import taichi.math as tm
import math

PI = math.pi

@ti.func
def t_lerp(val0, val1, scale):
    return (1 - scale) * val0 + scale * val1


@ti.func
def t_isNaN(val) -> bool:
    return tm.isnan(val)

@ti.func
def t_getBoxSD(pos: tm.vec3, center: tm.vec3, expand: tm.vec3, radius: float) -> float:
    dv = ti.abs(pos - center) - expand
    dav = ti.max(dv, tm.vec3(0, 0, 0))
    return ti.min(ti.max(ti.max(dv.x, dv.y), dv.z), 0.0) + dav.norm() - radius


@ti.func
def t_getCapsuleSD(pos: tm.vec3, center: tm.vec3, radius: float, half_length: float) -> float:
    a = center - tm.vec3(half_length, 0, 0)
    pa = pos - a
    ba = tm.vec3(2 * half_length, 0, 0)
    h = tm.clamp(tm.dot(pa, ba) / (4 * half_length * half_length), 0.0, 1.0)
    return (pa - ba * h).norm() - radius

