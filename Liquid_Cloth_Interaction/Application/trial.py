import taichi as ti
import taichi.math as tm

""" Use this file for DEBUG purcpose ONLY """

ti.init(arch=ti.cpu, debug=True)

v4 = tm.vec4(1,1,1,1)
mat = ti.Matrix.field(11, 4, dtype=float, shape=1)
ans = ti.Vector.field(11, dtype=float, shape=1)

@ti.kernel
def update():
    mat[0][0:3, 0] = tm.vec3(1,2,3)
    mat[0][1:4, 0] = tm.vec3(1,2,3)
    ans[0] = mat[0] @ v4

update()
print(ans[0])