create python3 set script
SET_RETURNS(x double, y double)
returns double as

def run(ctx):
    acc = 0.0
    while True:
        acc = acc + ctx.x + ctx.y
        if not ctx.next(): break
    return acc
/


create python3 set script
SET_EMITS(x double, y double)
emits (x double, y double) as

def run(ctx):
    while True:
        ctx.emit(ctx.y, ctx.x)
        if not ctx.next(): break
/


create python3 scalar script
SCALAR_RETURNS(x double, y double)
returns double as

def run(ctx):
    return ctx.x + ctx.y
/


create python3 scalar script
SCALAR_EMITS(x double, y double)
emits (x double, y double) as

def run(ctx):
    for i in range(int(ctx.x), int(ctx.y+1)):
        ctx.emit(float(i), float(i * i))
/
