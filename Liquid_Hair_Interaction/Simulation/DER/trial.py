import taichi as ti

ti.init(arch=ti.cpu)

x = ti.Vector.field(11, dtype=int, shape=2)

@ti.func
def prefix_sum(t0, t1):
    res = t0 + t1
    return res

@ti.kernel
def init():
    for i in x:
        x[i] = i + 1

@ti.kernel
def something():
    for i in range(1):
        for j in range(1, 10):
            x[j] = prefix_sum(x[j-1], x[j])

init()
res = x[0].outer_product(x[1])
print(res)
