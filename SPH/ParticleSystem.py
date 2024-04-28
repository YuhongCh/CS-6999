import taichi as ti
import taichi.math as tm

from Config import SPHConfig

EPSILON = 1e-6


class ParticleType:
    NonEmpty = -1
    Empty = 0
    Fluid = 1
    Boundary = 2


@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SPHConfig):
        # assign parameters from config
        self.max_particle_count = config.max_particle_count
        self.particle_radius = config.particle_radius
        self.particle_mass = config.particle_mass
        self.particle_volume = config.particle_volume
        self.support_radius = config.support_radius
        self.ref_density = config.ref_density

        # attribute to Taichi limit, use a field as int here to use in @ti.kernel
        # use num_particle as public variable in other file
        self._num_particle_field = ti.field(dtype=ti.int32, shape=())
        self.num_fluid_particle = 0
        self.num_solid_particle = 0

        # declare needed array fields
        self.position = ti.Vector.field(3, dtype=float, shape=self.max_particle_count)
        self.velocity = ti.Vector.field(3, dtype=float, shape=self.max_particle_count)
        self.acceleration = ti.Vector.field(3, dtype=float, shape=self.max_particle_count)
        self.type = ti.field(dtype=int, shape=self.max_particle_count)
        self.density = ti.field(float, shape=self.max_particle_count)

        # precomputed parameters
        self.inv_particle_mass = 1 / self.particle_mass

    @ti.kernel
    def __generate_particles(self, start_domain: tm.vec3, end_domain: tm.vec3, interval: float, particle_type: int) -> int:
        """
         generate particles compactly within the bounding box
         @return number of particles generated
        """
        dimension = ti.cast((end_domain - start_domain) / interval + 1, dtype=int)

        for coord in ti.grouped(ti.ndrange((0, dimension.x), (0, dimension.y), (0, dimension.z))):
            position = start_domain + coord * interval
            pid = ti.atomic_add(self._num_particle_field[None], 1)
            self.position[pid] = position
            self.type[pid] = particle_type
        return dimension.x * dimension.y * dimension.z

    def generate_boundary(self, start: tm.vec3, end: tm.vec3):
        interval = self.particle_radius * 2
        num_particle = 0
        num_particle += self.__generate_particles(start, tm.vec3(start.x, end.y, end.z), interval, ParticleType.Boundary)
        num_particle += self.__generate_particles(start, tm.vec3(end.x, start.y, end.z), interval, ParticleType.Boundary)
        num_particle += self.__generate_particles(start, tm.vec3(end.x, end.y, start.z), interval, ParticleType.Boundary)
        num_particle += self.__generate_particles(tm.vec3(start.x, start.y, end.z), tm.vec3(end.x, end.y, end.z), interval, ParticleType.Boundary)
        num_particle += self.__generate_particles(tm.vec3(end.x, start.y, start.z), tm.vec3(end.x, end.y, end.z), interval, ParticleType.Boundary)
        self.num_solid_particle += num_particle

    def generate_particles(self, start_domain: tm.vec3, end_domain: tm.vec3, particle_type: int):
        interval = self.support_radius
        num_particle = self.__generate_particles(start_domain, end_domain, interval, particle_type)
        if particle_type == ParticleType.Boundary:
            self.num_solid_particle += num_particle
        elif particle_type == ParticleType.Fluid:
            self.num_fluid_particle += num_particle

    @ti.kernel
    def reset_parameters(self):
        for pid in range(self._num_particle_field[None]):
            self.density[pid] = 0.0
            self.acceleration[pid] = tm.vec3(0, 0, 0)

    def add_particle(self, position: tm.vec3, particle_type: int):
        self.position[self._num_particle_field[None]] = position
        self.type[self._num_particle_field[None]] = particle_type
        self._num_particle_field[None] += 1
        if particle_type == ParticleType.Boundary:
            self.num_solid_particle += 1
        elif particle_type == ParticleType.Fluid:
            self.num_fluid_particle += 1

    def get_particle_num(self, particle_type: int) -> int:
        """ -1 for total number of particle added """
        if particle_type == ParticleType.NonEmpty:
            return self._num_particle_field[None]
        elif particle_type == ParticleType.Empty:
            return self.max_particle_count - self._num_particle_field[None]
        elif particle_type == ParticleType.Fluid:
            return self.num_fluid_particle
        elif particle_type == ParticleType.Boundary:
            return self.num_solid_particle
        else:
            raise NotImplementedError
