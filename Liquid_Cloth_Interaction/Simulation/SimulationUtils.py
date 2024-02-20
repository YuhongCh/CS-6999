import taichi as ti
import taichi.math as tm

""" Common Utils Variables Defined Below """
Epsilon = 1e-6


""" Common Utils Functions Defined Below """
"""
!!! Warning !!!
Be care of using methods here in Taichi-Scope methods.
Using function with return statement in Taichi-Scope loop is FORBIDDEN!
"""

""" Below are Python-Scope Methods"""


def p_find_normal(dir: tm.vec3) -> tm.vec3:
    """ Find the normal of given vector dir """
    if abs(dir.x) > Epsilon:
        return tm.vec3(-dir.y, dir.x, 0)
    elif abs(dir.y) > Epsilon:
        return tm.vec3(0, -dir.z, dir.y)
    elif abs(dir.z) > Epsilon:
        return tm.vec3(dir.z, 0, -dir.x)
    else:
        print("Error: Cannot find normal of a zero vector")
        assert False


""" Below are Taichi-Scope Methods """

@ti.func
def t_find_normal(dir: tm.vec3) -> tm.vec3:
    """ Find the normal of given vector dir """
    if abs(dir.x) > Epsilon:
        return tm.vec3(-dir.y, dir.x, 0)
    elif abs(dir.y) > Epsilon:
        return tm.vec3(0, -dir.z, dir.y)
    elif abs(dir.z) > Epsilon:
        return tm.vec3(dir.z, 0, -dir.x)
    else:
        print("Error: Cannot find normal of a zero vector")
        assert False
