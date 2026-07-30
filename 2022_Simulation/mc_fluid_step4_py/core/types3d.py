import random as pyrandom
import math

class Vec3:
    __slots__ = ('x','y','z')
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x); self.y = float(y); self.z = float(z)
    def __add__(self, o): return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o): return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def __mul__(self, s): return Vec3(self.x*s, self.y*s, self.z*s)
    def __rmul__(self, s): return Vec3(self.x*s, self.y*s, self.z*s)
    def __neg__(self): return Vec3(-self.x, -self.y, -self.z)
    def dot(self, o): return self.x*o.x + self.y*o.y + self.z*o.z
    def norm2(self): return self.x**2 + self.y**2 + self.z**2
    def norm(self): return math.sqrt(self.norm2())
    def cross(self, o): return Vec3(self.y*o.z - self.z*o.y,
                                    self.z*o.x - self.x*o.z,
                                    self.x*o.y - self.y*o.x)

class RNG3D:
    def __init__(self):
        self._uni = pyrandom.uniform
        self._gauss_cache = None

    def uniform(self, a=0.0, b=1.0):
        return self._uni(a, b)

    def gaussian(self, mean=0.0, std=1.0):
        """Box‑Muller 生成标准正态分布（使用缓存）"""
        if self._gauss_cache is not None:
            val = self._gauss_cache
            self._gauss_cache = None
            return mean + std * val
        else:
            u1 = self.uniform()
            u2 = self.uniform()
            r = math.sqrt(-2 * math.log(u1))
            self._gauss_cache = r * math.sin(2 * math.pi * u2)
            return mean + std * (r * math.cos(2 * math.pi * u2))