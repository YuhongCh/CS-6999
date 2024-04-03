import taichi as ti
import taichi.math as tm

EPSILON = 1e-12

@ti.func
def t_isSmall(val: float) -> bool:
    return abs(val) < EPSILON

@ti.func
def t_isUnit(v: tm.vec3) -> bool:
    return t_isSmall(tm.dot(v, v) - 1.0)

@ti.func
def t_getOrthonormal(v: tm.vec3, n: tm.vec3) -> tm.vec3:
    res = v - tm.dot(v, n) * n
    return tm.nomalize(res)

@ti.func
def t_getParallelTransport(u: tm.vec3, t0: tm.vec3, t1: tm.vec3) -> tm.vec3:
    t0ct1 = tm.cross(t0, t1)
    t0ct1_norm = tm.length(t0ct1)

    res = u
    if t0ct1_norm > EPSILON:
        t0ct1 /= t0ct1_norm
        n0 = tm.normalize(tm.cross(t0, t0ct1))
        n1 = tm.normalize(tm.cross(t1, t0ct1))
        t1_norm = tm.normalize(t1)
        t0_norm = tm.normalize(t0)
        res = tm.dot(u, t0_norm) * t1_norm
        res += tm.dot(u, n0) * n1 + tm.dot(u, t0ct1) * t0ct1
    return res

@ti.func
def t_getNormalParallelTransport(u: tm.vec3, t0: tm.vec3, t1: tm.vec3) -> tm.vec3:
    t0ct1 = tm.cross(t0, t1)
    t0ct1_norm = tm.length(t0ct1)

    res = u
    if t0ct1_norm > EPSILON:
        t0ct1 /= t0ct1_norm
        n0 = tm.normalize(tm.cross(t0, t0ct1))
        n1 = tm.normalize(tm.cross(t1, t0ct1))
        res = tm.dot(u, n0) * n1 + tm.dot(u, t0ct1) * t0ct1
    return res

@ti.func
def t_getOrthonormalParallelTransport(u: tm.vec3, t0: tm.vec3, t1: tm.vec3) -> tm.vec3:
    t0ct1 = tm.cross(t0, t1)
    t0ct1_norm = tm.length(t0ct1)

    res = u
    if t0ct1_norm > EPSILON:
        t0ct1 /= t0ct1_norm
        n0 = tm.cross(t0, t0ct1)
        n1 = tm.cross(t1, t0ct1)
        res = tm.dot(u, n0) * n1 + tm.dot(u, t0ct1) * t0ct1
    return res

@ti.func
def t_getRotateAxisAngle(v: tm.vec3, axis: tm.vec3, angle: float) -> tm.vec3:
    assert(t_isUnit(axis))
    res = v
    if angle > EPSILON:
        cos_val = tm.cos(angle)
        sin_val = tm.sin(angle)
        res = cos_val * v + sin_val * tm.cross(angle, v) + tm.dot(axis, v) * (1.0 - cos_val) * axis
    return res

@ti.func
def t_getSignedAngle(u: tm.vec3, v: tm.vec3, n: tm.vec3) -> tm.vec3:
    w = tm.cross(u, v)
    angle = tm.atan2(tm.length(w), tm.dot(u, v))

    if tm.dot(n, w) < 0:
        angle = -angle
    return angle

@ti.func
def t_getCrossMatrix(v: tm.vec3) -> tm.mat3:
    return tm.mat3(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)


