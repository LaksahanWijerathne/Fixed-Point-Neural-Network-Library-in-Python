# fixed_point_arithmetic.py

class FixedPointArithmetic:
    def __init__(self, decimal_points):
        self.scaling_factor = 10 ** decimal_points

    def scale(self, value):
        return int(value * self.scaling_factor)

    def descale(self, value):
        return value / self.scaling_factor
