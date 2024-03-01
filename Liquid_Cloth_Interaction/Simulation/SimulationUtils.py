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
    normal = tm.vec3(0, 0, 0)
    if abs(dir[0]) > Epsilon:
        normal = tm.vec3(-dir[1], dir[0], 0)
    elif abs(dir[1]) > Epsilon:
        normal = tm.vec3(0, -dir[0], dir[1])
    elif abs(dir[2]) > Epsilon:
        normal = tm.vec3(dir[2], 0, -dir[0])
    else:
        print("Error: Cannot find normal of a zero vector")
    return normal


""" Below are Taichi-Scope Methods """

@ti.func
def t_find_normal(dir: tm.vec3) -> tm.vec3:
    """ Find the normal of given vector dir """
    normal = tm.vec3(0, 0, 0)
    if abs(dir[0]) > Epsilon:
        normal = tm.vec3(-dir[1], dir[0], 0)
    elif abs(dir[1]) > Epsilon:
        normal = tm.vec3(0, -dir[0], dir[1])
    elif abs(dir[2]) > Epsilon:
        normal = tm.vec3(dir[2], 0, -dir[0])
    else:
        print("Error: Cannot find normal of a zero vector")
    return normal