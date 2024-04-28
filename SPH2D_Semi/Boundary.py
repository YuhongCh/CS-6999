import taichi as ti
import taichi.math as tm

import math

from Kernel import CubicSplineKernel
from Config import SPHConfig

@ti.data_oriented
class Boundary:
    def __init__(self, center: tm.vec2, expand: tm.vec2, radius: float, config: SPHConfig):
        # Ideally, center = (0.5, 0.5), expand = (0.5, 0.5), radius = 0
        self.pdf = ti.pow(config.particle_radius / (config.support_radius + config.particle_radius), 2)
        self.particle_radius = config.particle_radius
        self.support_radius = config.support_radius
        self.ref_density = config.ref_density

        self.center = center
        self.expand = expand
        self.radius = radius
        self.kernel = CubicSplineKernel()
        print(center, expand, radius, self.pdf)

    @ti.func
    def get_sdf(self, position: tm.vec2) -> float:
        """ SDF for 2D box """
        dx, dy = ti.abs(position - self.center) - self.expand
        dax, day = ti.max(dx, 0.0), ti.max(dy, 0.0)
        return ti.min(ti.max(dx, dy), 0.0) + tm.sqrt(dax * dax + day * day) - self.radius

    @ti.func
    def get_evaluation(self, sd: float) -> float:
        result = 0.0
        if sd <= self.support_radius:
            result = self.ref_density * (1 - sd / self.support_radius)
        return result

    @ti.func
    def get_boundary_density(self, pos: tm.vec2) -> float:
        # sample the points
        num_sample = 100
        sum = 0.0
        num_success = 0

        for i in range(num_sample):
            r = self.support_radius * ti.sqrt(ti.random())
            theta = ti.random() * 2 * math.pi
            npos = tm.vec2(pos.x + r * tm.cos(theta), pos.y + r * tm.sin(theta))
            evaluate_result = self.get_evaluation(-self.get_sdf(npos))
            if ti.abs(evaluate_result) > 1e-6:
                num_success += 1
                sum += evaluate_result * self.kernel.get_weight(pos - npos, self.support_radius)
        return sum / self.pdf / num_sample
