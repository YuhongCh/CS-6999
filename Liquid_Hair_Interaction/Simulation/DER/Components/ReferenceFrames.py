import taichi as ti
import taichi.math as tm

import Liquid_Hair_Interaction.Simulation.DER.Utils as Utils
from Liquid_Hair_Interaction.Simulation.DER.Components.DOFs import Tangents


@ti.data_oriented
class ReferenceFrames1:
    def __init__(self, tangents: Tangents):
        self.tangents = tangents
        self.prev_tangents_data = ti.Vector.field(3, dtype=float, shape=tangents.size)

        self.start_index = 0
        self.end_index = self.tangents.size
        self.size = self.tangents.size

        self.data = ti.Vector.field(3, dtype=float, shape=self.size)

    @ti.kernel
    def t_setInitFrame(self, init_frame: tm.vec3):
        # invalid initial frame
        init_frame = tm.normalize(init_frame)
        if not Utils.t_isSmall(abs(tm.dot(init_frame, init_frame) - 1)):
            assert(False)

        """ This is a space-based parallel transport """
        # An outer loop to prevent parallelism
        for tmp in range(1):
            for idx in range(1, self.end_index):
                result = Utils.t_getOrthonormalParallelTransport(self.data[idx - 1], 
                                                                self.tangents.data[idx - 1], 
                                                                self.tangents.data[idx])
                result = Utils.t_getOrthonormal(result, self.tangents.data[idx])
                self.data[idx] = result
                self.prev_tangents_data[idx] = self.tangents.data[idx]

    @ti.kernel
    def t_compute(self):
        """ This is a time-based parallel transport """
        for idx in range(self.start_index, self.end_index):
            prev_trangent = self.prev_tangents_data[idx]
            curr_tangent = self.tangents.data[idx]
            result = Utils.t_getOrthonormalParallelTransport(self.data[idx], prev_trangent, curr_tangent)
            result = Utils.t_getOrthonormal(result, curr_tangent)
            self.data[idx] = result
            self.prev_tangents_data[idx] = self.tangents.data[idx]

    @ti.func
    def t_isNormal(self) -> bool:
        is_normal = True
        for idx in range(self.start_index, self.end_index):
            if not Utils.t_isSmall(tm.dot(self.data[idx], self.tangents.data[idx])):
                is_normal = False
                break
        return is_normal
    
    
@ti.data_oriented
class ReferenceFrames2:
    def __init__(self, tangents: Tangents, ref_frame1: ReferenceFrames1):
        self.tangents = tangents
        self.ref_frame1 = ref_frame1

        self.start_index = 0
        self.end_index = self.tangents.size
        self.size = self.tangents.size

        self.data = ti.Vector.field(3, dtype=float, shape=self.size)

    @ti.kernel
    def t_compute(self):
        """ This is a time-based parallel transport """
        for idx in range(self.start_index, self.end_index):
            curr_tangent = self.tangents.data[idx]
            curr_ref_frame1 = self.ref_frame1.data[idx]
            self.data[idx] = tm.cross(curr_tangent, curr_ref_frame1)


@ti.data_oriented
class ReferenceTwists:
    def __init__(self, tangents: Tangents, ref_frame1: ReferenceFrames1):
        self.tangents = tangents
        self.ref_frame1 = ref_frame1

        self.start_index = 1
        self.end_index = self.tangents.size
        self.size = self.tangents.size
        self.data = ti.field(dtype=float, shape=self.size)

    @ti.func
    def t_compute(self):
        for idx in range(self.start_index, self.end_index):
            u0 = self.ref_frame1.data[idx - 1]
            u1 = self.ref_frame1.data[idx]
            prev_rangent = self.tangents.data[idx - 1]
            curr_tangent = self.tangents.data[idx]
            ut = Utils.t_getOrthonormalParallelTransport(u0, prev_rangent, curr_tangent)

            prev_twist = self.data[idx]
            ut = Utils.t_getRotateAxisAngle(ut, curr_tangent, prev_twist)
            self.data[idx] = prev_twist + Utils.t_getSignedAngle(ut, u1, curr_tangent)
