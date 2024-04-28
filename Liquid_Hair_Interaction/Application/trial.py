import taichi as ti
import taichi.math as tm

""" Use this file to tryout simple taichi functions ONLY """

ti.init(arch=ti.cpu, debug=True)

v4 = tm.vec4(1,2,3,4)
mat = ti.Matrix.field(3, 3, dtype=float, shape=1)
ans = ti.Vector.field(11, dtype=float, shape=1)

@ti.kernel
def update():
    id = tm.mat3(1, 1, 1, 1, 1, 1, 1, 1, 1)
    mat[0] = id

update()
print(mat[0])