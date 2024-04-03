import taichi as ti

from Liquid_Hair_Interaction.Simulation.DER.Components.ElasticParameters import *
from Liquid_Hair_Interaction.Simulation.DER.Components.Kappas import *

@ti.data_oriented
class BendingProducts:
    def __init__(self, bending_matrix: BaseBendingMatrix, grad_kappas: GradKappas):
       self.bending_matrix = bending_matrix
       self.kappas = grad_kappas

       self.start_index = 1
       self.end_index = self.grad_kappas.size
       self.size = self.grad_kappas.size
       self.data = ti.Matrix.field(11, 11, dtype=float, shape=self.size)

    @ti.kernel
    def t_compute(self):
       for idx in range(self.start_index, self.end_index):
          # compute KBK^T
          self.data[idx] = self.grad_kappas.data[idx] @ self.bending_matrix.data @ self.grad_kappas.data[idx].transpose()

