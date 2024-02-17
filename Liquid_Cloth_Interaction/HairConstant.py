"""
TODO: After complete basic trial, remove this file and use real test for future work
"""

from math import pi

""" Below are scene parameters """
dt = 0.001
Epsilon = 0.0000001

""" Below are hair parameters """
num_hair_strand = 1
num_hair_vertices = 5
hair_mass = 1
hair_radius = 0.018
hair_length = 1

Youngs_Modulus = 6.6e7
Shear_Modulus = 0.365 * Youngs_Modulus

# pre-computed
hair_ref_length = hair_length / num_hair_vertices
hair_ks = Youngs_Modulus * pi * hair_radius * hair_radius
