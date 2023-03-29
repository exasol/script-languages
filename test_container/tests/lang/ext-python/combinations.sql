create external set script
SET_RETURNS(x double, y double)
returns double AS
# redirector @@redirector_url@@

def run(ctx):
    acc = 0.0
    while True:
        acc = acc + ctx.x + ctx.y
        if not ctx.next(): break
    return acc
/


create external set script
SET_EMITS(x double, y double)
emits (x double, y double) AS
# redirector @@redirector_url@@

def run(ctx):
    while True:
        ctx.emit(ctx.y, ctx.x)
        if not ctx.next(): break
/


create external scalar script
SCALAR_RETURNS(x double, y double)
returns double AS
# redirector @@redirector_url@@

def run(ctx):
    return ctx.x + ctx.y
/


create external scalar script
SCALAR_EMITS(x double, y double)
emits (x double, y double) AS
# redirector @@redirector_url@@

def run(ctx):
    for i in range(int(ctx.x), int(ctx.y+1)):
        ctx.emit(float(i), float(i * i))
/
