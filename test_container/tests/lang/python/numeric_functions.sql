CREATE python SCALAR SCRIPT
pi()
RETURNS double AS

import math

def run(ctx):
  return math.pi
/

CREATE python SCALAR SCRIPT
double_mult("x" double, "y" double)
RETURNS double AS

def run(ctx):
  if ctx.x is None or ctx.y is None:
    return None
  else:
    return ctx.x * ctx.y
/

CREATE python SCALAR SCRIPT add_two_doubles(x DOUBLE, y DOUBLE) RETURNS DOUBLE AS
def run(ctx):
    if ctx.x is not None and ctx.y is not None:
        return ctx.x + ctx.y;
    return None
/

CREATE python SCALAR SCRIPT
add_three_doubles(x DOUBLE, y DOUBLE, z DOUBLE)
RETURNS DOUBLE AS
def run(ctx):
    if ctx.x is not None and ctx.y is not None and ctx.z is not None:
        return ctx.x + ctx.y + ctx.z;
    return None
/

CREATE python SCALAR SCRIPT
split_integer_into_digits("x" INTEGER)
EMITS (y INTEGER) AS
import math
def run(ctx):
    if ctx.x is not None:
        y = abs(ctx.x)
        while y > 0:
            ctx.emit(y % 10)
            y //= 10
/


-- vim: ts=2:sts=2:sw=2:et
