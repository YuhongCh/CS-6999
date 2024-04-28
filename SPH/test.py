import taichi as ti

ti.init(arch=ti.cpu)

for delta_x, delta_y, delta_z in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
    print(delta_x, delta_y, delta_z)