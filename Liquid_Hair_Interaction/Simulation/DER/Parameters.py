import taichi as ti
import math

from Liquid_Hair_Interaction.Simulation.DER.Components.ElasticParameters import *


@ti.data_oriented
class StrandEquilibriumParameters:
    def __init__(self, vertices, curl_radius, curl_density, dL, root_length, valid):
        self.vertices = vertices
        self.curl_radius = curl_radius
        self.curl_density = curl_density
        self.dL = dL
        self.root_length = root_length
        self.valid = valid
        self.is_dirty = False


@ti.data_oriented
class StrandParameters:
    def __init__(self, radius, Youngs_Modulus, Shear_Modulus, stretching_multiplier, density, viscosity,
                 base_rotation, dt, accumulate_viscousity, accumulate_viscousity_bend, variable_radius_hair,
                 straight_hairs, color):
        self.density = density
        self.viscosity = viscosity

        self.viscous_bending_coef = 0.0
        self.viscousKt = 0.0
        self.viscousKs = 0.0

        self.radius = radius
        self.base_rotation = base_rotation
        self.base_bending_matrix = BaseBendingMatrix(radius, base_rotation)
        self.Youngs_Modulus = Youngs_Modulus
        self.Shear_modulus = Shear_Modulus
        self.stretching_multiplier = stretching_multiplier
        self.elastic_ks = ElasticKs(radius, Youngs_Modulus)
        self.elastic_kt = ElasticKt(radius, Shear_Modulus)
        self.color = color
        self.accumulate_viscousity = accumulate_viscousity
        self.accumulate_viscousity_bend = accumulate_viscousity_bend
        self.variable_hair_radius = variable_radius_hair
        self.straight_hairs = straight_hairs
        self.computeViscousForceCoefficients(dt)

    def computeViscousForceCoefficients(self, dt):
      self.viscousKs = 3 * math.pi * self.radius * self.viscosity / dt
      self.viscousKt = 2 * ti.pow(math.pi, 4) * ti.pow(self.radius, 4) * self.viscosity / dt
      self.viscous_bending_coef = 3 * self.viscosity / dt

    @ti.func
    def t_getRadiusMultiplier(self, vertex_idx: int, num_vertices: int) -> float:
        multiplier = 1.0
        if self.variable_radius_hair:
            s = vertex_idx / (num_vertices - 1.0)
            multiplier = ti.exp(-3.4612 * s) * self.straight_hairs + (1.0 - self.straight_hairs)
        return multiplier
    
    @ti.func
    def t_getKs(self, vertex_idx: int, num_vertices: int) -> float:
        t = self.t_getRadiusMultiplier(vertex_idx, num_vertices)
        return t * t * self.elastic_ks.ks * self.stretching_multiplier
    
    @ti.func
    def t_getKt(self, vertex_idx: int, num_vertices: int) -> float:
        t = self.t_getRadiusMultiplier(vertex_idx, num_vertices)
        return ti.pow(t, 4) * self.elastic_kt.kt
    
    @ti.func
    def t_getRadius(self, vertex_idx: int, num_vertices: int) -> float:
        t = self.t_getRadiusMultiplier(vertex_idx, num_vertices)
        return t * self.radius
    
    @ti.func
    def t_getBendingCoefficient(self, vertex_idx: int, num_vertices: int) -> float:
        t = self.t_getRadiusMultiplier(vertex_idx, num_vertices)
        return ti.pow(t, 4) * self.Youngs_Modulus
    
    @ti.func
    def t_getBendingCoefficient(self, vertex_idx: int, num_vertices: int) -> float:
        t = self.t_getRadiusMultiplier(vertex_idx, num_vertices)
        return ti.pow(t, 4) * self.Youngs_Modulus

    @ti.func
    def t_getBendingMatrix(self, vertex_idx: int, num_vertices: int) -> tm.mat2:
        coef = self.t_getBendingCoefficient(vertex_idx, num_vertices)
        return coef * self.base_bending_matrix.data
    
    @ti.func
    def t_getViscousBendingCoefficient(self, vertex_idx: int, num_vertices: int) -> float:
      t = self.t_getRadiusMultiplier(vertex_idx, num_vertices)
      return ti.pow(t, 4) * self.viscous_bending_coef
    
    @ti.func
    def t_getViscousBendingMatrix(self, vertex_idx: int, num_vertices: int) -> tm.mat2:
        return self.t_getViscousBendingCoefficient(vertex_idx, num_vertices) * self.base_bending_matrix.data
    
    @ti.func
    def t_getViscousKs(self, vertex_idx: int, num_vertices: int) -> float:
        t = self.t_getRadiusMultiplier(vertex_idx, num_vertices) 
        return ti.pow(t, 2) * self.viscousKs * self.stretching_multiplier
    
    @ti.func
    def t_getViscousKt(self, vertex_idx: int, num_vertices: int) -> float:
        t = self.t_getRadiusMultiplier(vertex_idx, num_vertices) 
        return ti.pow(t, 4) * self.viscousKt
