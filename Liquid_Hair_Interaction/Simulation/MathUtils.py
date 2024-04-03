import taichi as ti
import taichi.math as tm
import math

PI = math.pi

@ti.func
def t_lerp(val0, val1, scale):
  return (1 - scale) * val0 + scale * val1

@ti.func
def t_isNaN(val) -> bool:
  tm.nan