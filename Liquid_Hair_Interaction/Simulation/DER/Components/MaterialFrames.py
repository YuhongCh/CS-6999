import taichi as ti
import taichi.math as tm

from Liquid_Hair_Interaction.Simulation.DER.Components.DOFs import TrigThetas
from Liquid_Hair_Interaction.Simulation.DER.Components.ReferenceFrames import ReferenceFrames1, ReferenceFrames2


@ti.data_oriented
class MaterialFrames:
    def __init__(self, trig_thetas: TrigThetas, ref_frame1: ReferenceFrames1, ref_frame2: ReferenceFrames2):
        self.trig_thetas = trig_thetas
        self.ref_frame1 = ref_frame1
        self.ref_frame2 = ref_frame2

        self.start_index = 0
        self.end_index = self.ref_frame1.size
        self.size = self.ref_frame1.size
        self.data = ti.Vector.field(3, dtype=float, shape=self.size)

    @ti.func
    def t_getLinearMix(self, u: tm.vec3, v: tm.vec3, s: float, c: float) -> tm.vec3:
        raise NotImplementedError

    @ti.kernel
    def t_compute(self):
        for idx in range(self.start_index, self.end_index):
            rf1 = self.ref_frame1.data[idx]
            rf2 = self.ref_frame2.data[idx]
            s = self.trig_thetas.sin_data[idx]
            c = self.trig_thetas.cos_data[idx]
            self.data[idx] = self.t_getLinearMix(rf1, rf2, s, c)


@ti.data_oriented
class MaterialFrames1(MaterialFrames):
    def __init__(self, trig_thetas: TrigThetas, ref_frame1: ReferenceFrames1, ref_frame2: ReferenceFrames2):
        super().__init__(trig_thetas, ref_frame1, ref_frame2)

    @ti.func
    def t_getLinearMix(self, u: tm.vec3, v: tm.vec3, s: float, c: float) -> tm.vec3:
        return u * c + s * v
    
@ti.data_oriented
class MaterialFrames2(MaterialFrames):
    def __init__(self, trig_thetas: TrigThetas, ref_frame1: ReferenceFrames1, ref_frame2: ReferenceFrames2):
        super().__init__(trig_thetas, ref_frame1, ref_frame2)

    @ti.func
    def t_getLinearMix(self, u: tm.vec3, v: tm.vec3, s: float, c: float) -> tm.vec3:
        return -s * u + c * v
