import numpy as np
import random as pyrandom

class Vec2:
    __slots__ = ('x', 'y')
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x); self.y = float(y)
    def __add__(self, other): return Vec2(self.x+other.x, self.y+other.y)
    def __sub__(self, other): return Vec2(self.x-other.x, self.y-other.y)
    def __mul__(self, s):    return Vec2(self.x*s, self.y*s)
    def __rmul__(self, s):   return Vec2(self.x*s, self.y*s)    # <-- 新增
    def __neg__(self):       return Vec2(-self.x, -self.y)      # <-- 新增
    def __truediv__(self, s):return Vec2(self.x/s, self.y/s)    # 可选
    def dot(self, other):    return self.x*other.x + self.y*other.y
    def norm2(self):         return self.x**2 + self.y**2
    def norm(self):          return np.sqrt(self.norm2())

def perp(v): return Vec2(-v.y, v.x)

class RNG:
    def __init__(self):
        self.uni = pyrandom.uniform   # 保留原有对象引用（可选）
    def uniform(self, a=0.0, b=1.0) -> float:
        """ 返回 [a, b) 内的均匀随机浮点数 """
        return pyrandom.uniform(a, b)
    def uniform_in_box(self, xmin, xmax, ymin, ymax) -> Vec2:
        return Vec2(self.uniform(xmin, xmax), self.uniform(ymin, ymax))