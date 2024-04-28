import taichi as ti
import taichi.math as tm


@ti.data_oriented
class GravityForce:
    def __init__(self, force_object):
        self.force_object = force_object

    @ti.kernel
    def accumulateForce(self):
        for i in self.force_object.force:
            self.force_object.force[i] += tm.vec3(0, -9.81, 0)
