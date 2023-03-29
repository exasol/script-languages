--CREATE <lang>  SCALAR SCRIPT
--base_pi()
--RETURNS DOUBLE AS

-- pi

CREATE EXTERNAL SCALAR SCRIPT
basic_emit_two_ints()
EMITS (i INTEGER, j INTEGER) AS
# redirector @@redirector_url@@
def run(ctx):
	ctx.emit(1,2)
/

CREATE EXTERNAL SCALAR SCRIPT
basic_nth_partial_sum(n INTEGER)
RETURNS INTEGER as
# redirector @@redirector_url@@

def run(ctx):
    if ctx.n is not None:
        return ctx.n * (ctx.n + 1) / 2
    return 0
/


CREATE EXTERNAL SCALAR SCRIPT
basic_range(n INTEGER)
EMITS (n INTEGER) AS
# redirector @@redirector_url@@

def run(ctx):
    if ctx.n is not None:
        for i in range(ctx.n):
            ctx.emit(i)
/

CREATE EXTERNAL SET SCRIPT
basic_sum(x INTEGER)
RETURNS INTEGER AS
# redirector @@redirector_url@@

def run(ctx):
    s = 0
    while True:
        if ctx.x is not None:
            s += ctx.x
        if not ctx.next():
            break
    return s
/

CREATE EXTERNAL SET SCRIPT
basic_sum_grp(x INTEGER)
EMITS (s INTEGER) AS
# redirector @@redirector_url@@

def run(ctx):
    s = 0
    while True:
        if ctx.x is not None:
            s += ctx.x
        if not ctx.next():
            break
    ctx.emit(s)
/


CREATE EXTERNAL SET SCRIPT
set_returns_has_empty_input(a double) RETURNS boolean AS
# redirector @@redirector_url@@

def run(ctx):
    return bool(ctx.x is None)
/

CREATE EXTERNAL SET SCRIPT
set_emits_has_empty_input(a double) EMITS (x double, y varchar(10)) AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is None:
        ctx.emit(1,'1')
    else:
        ctx.emit(2,'2')
/

-- vim: ts=4:sts=4:sw=4
