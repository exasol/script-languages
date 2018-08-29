--CREATE <lang>  SCALAR SCRIPT
--base_pi()
--RETURNS DOUBLE AS

-- pi

CREATE python SCALAR SCRIPT
basic_emit_several_groups(a INTEGER, b INTEGER)
EMITS (i INTEGER, j VARCHAR(40)) AS
def run(ctx):
    for n in range(ctx.a):
        for i in range(ctx.b):
            ctx.emit(i, repr((exa.meta.vm_id, exa.meta.node_count, exa.meta.node_id)))
/

CREATE python SET SCRIPT
basic_test_reset(i INTEGER, j VARCHAR(40))
EMITS (k INTEGER) AS
def run(ctx):
    ctx.emit(ctx.i)
    ctx.next()
    ctx.emit(ctx.i)
    ctx.reset()
    ctx.emit(ctx.i)
    ctx.next()
    ctx.emit(ctx.i)
/

CREATE python SCALAR SCRIPT
basic_emit_two_ints()
EMITS (i INTEGER, j INTEGER) AS
def run(ctx):
	ctx.emit(1,2)
/

CREATE python SCALAR SCRIPT
basic_nth_partial_sum(n INTEGER)
RETURNS INTEGER as

def run(ctx):
    if ctx.n is not None:
        return ctx.n * (ctx.n + 1) / 2
    return 0
/


CREATE python SCALAR SCRIPT
basic_range(n INTEGER)
EMITS (n INTEGER) AS

def run(ctx):
    if ctx.n is not None:
        for i in range(ctx.n):
            ctx.emit(i)
/

CREATE python SET SCRIPT
basic_sum(x INTEGER)
RETURNS INTEGER AS

def run(ctx):
    s = 0
    while True:
        if ctx.x is not None:
            s += ctx.x
        if not ctx.next():
            break
    return s
/

CREATE python SET SCRIPT
basic_sum_grp(x INTEGER)
EMITS (s INTEGER) AS

def run(ctx):
    s = 0
    while True:
        if ctx.x is not None:
            s += ctx.x
        if not ctx.next():
            break
    ctx.emit(s)
/


CREATE python SET SCRIPT
set_returns_has_empty_input(a double) RETURNS boolean AS

def run(ctx):
    return bool(ctx.x is None)
/

CREATE python SET SCRIPT
set_emits_has_empty_input(a double) EMITS (x double, y varchar(10)) AS
def run(ctx):
    if ctx.x is None:
        ctx.emit(1,'1')
    else:
        ctx.emit(2,'2')
/

-- vim: ts=4:sts=4:sw=4
