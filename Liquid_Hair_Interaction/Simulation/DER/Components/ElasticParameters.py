import taichi as ti
import taichi.math as tm
import math


@ti.data_oriented
class BaseBendingMatrix:
    def __init__(self, physical_radius, base_rotation):
       self.physical_radius = physical_radius
       self.base_rotation = base_rotation
       self.data = tm.mat2(1, 0, 0, 1)

    @ti.func
    def compute(self):
        """ For simplicity, I now use identity matrix, implying no bending """
        self.data = tm.mat2(1, 0, 0, 1)


@ti.data_oriented
class ElasticKs:
   def __init__(self, physical_radius, Youngs_Modulus):
        self.physical_radius = physical_radius
        self.Youngs_Modulus = Youngs_Modulus
        self.ks = 1.0

   @ti.func
   def compute(self):
        self.ks = math.pi * self.physical_radius * self.physical_radius * self.Youngs_Modulus


@ti.data_oriented
class ElasticKt:
   def __init__(self, physical_radius, Shear_Modulus):
        self.physical_radius = physical_radius
        self.Shear_Modulus = Shear_Modulus
        self.kt = 1.0
    
   @ti.func
   def compute(self):
        self.kt = 2.0 * math.pi * ti.pow(self.physical_radius, 4) * self.Shear_Modulus

