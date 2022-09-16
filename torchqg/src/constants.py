import math
from dataclasses import dataclass

@dataclass
class Units:
    
    def unitless(x):
        raise NotImplementedError()
        
    def real(x):
        raise NotImplementedError()
        
        
        
class Time(Units):
    conversion: float = 1.2e6
    
    def unitless(x):
        return x * self.conversion
    
    def real(x):
        return x / conversion
    


def t_unit():
    return 1.2e6

def l_unit():
    return (504e4 / math.pi)


def time_r2u(t):
    return t / t_unit()

def time_u2r(t):
    return t * t_unit()

# def space_2_unitless(