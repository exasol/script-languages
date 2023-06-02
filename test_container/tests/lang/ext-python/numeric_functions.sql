create external SCALAR SCRIPT
pi()
RETURNS double AS
# redirector @@redirector_url@@

import math

def run(ctx):
  return math.pi
/

create external SCALAR SCRIPT
double_mult("x" double, "y" double)
RETURNS double AS
# redirector @@redirector_url@@

def run(ctx):
  if ctx.x is None or ctx.y is None:
    return None
  else:
    return ctx.x * ctx.y
/

create external SCALAR SCRIPT add_two_doubles(x DOUBLE, y DOUBLE) RETURNS DOUBLE AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is not None and ctx.y is not None:
        return ctx.x + ctx.y;
    return None
/

create external SCALAR SCRIPT
add_three_doubles(x DOUBLE, y DOUBLE, z DOUBLE)
RETURNS DOUBLE AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is not None and ctx.y is not None and ctx.z is not None:
        return ctx.x + ctx.y + ctx.z;
    return None
/

create external SCALAR SCRIPT
split_integer_into_digits("x" INTEGER)
EMITS (y INTEGER) AS
# redirector @@redirector_url@@
import math
def run(ctx):
    if ctx.x is not None:
        y = abs(ctx.x)
        while y > 0:
            ctx.emit(y % 10)
            y //= 10
/


-- vim: ts=2:sts=2:sw=2:et
