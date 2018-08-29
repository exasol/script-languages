create external SCALAR SCRIPT echo_boolean(x BOOLEAN) RETURNS BOOLEAN AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is not None:
        return ctx.x
    return None
/

create external SCALAR SCRIPT echo_char1(x CHAR(1)) RETURNS CHAR(1) AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is not None:
        return ctx.x
    return None
/

create external SCALAR SCRIPT echo_char10(x CHAR(10)) RETURNS CHAR(10) AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is not None:
        if len(ctx.x) != 10:
            return None
        else:
            return ctx.x
    return None
/

create external SCALAR SCRIPT echo_date(x DATE) RETURNS DATE AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is not None:
        return ctx.x
    return None
/

create external SCALAR SCRIPT echo_integer(x INTEGER) RETURNS INTEGER AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is not None:
        return ctx.x
    return None
/

create external SCALAR SCRIPT echo_double(x DOUBLE) RETURNS DOUBLE AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is not None:
        return ctx.x
    return None
/

create external SCALAR SCRIPT echo_decimal_36_0(x DECIMAL(36,0)) RETURNS DECIMAL(36,0) AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is not None:
        return ctx.x
    return None
/

create external SCALAR SCRIPT echo_decimal_36_36(x DECIMAL(36,36)) RETURNS DECIMAL(36,36) AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is not None:
        return ctx.x
    return None
/


create external SCALAR SCRIPT echo_varchar10(x VARCHAR(10)) RETURNS VARCHAR(10) AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is not None:
        return ctx.x
    return None
/

create external SCALAR SCRIPT echo_timestamp(x TIMESTAMP) RETURNS TIMESTAMP AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.x is not None:
        return ctx.x
    return None
/

create external SCALAR SCRIPT run_func_is_empty() RETURNS DOUBLE AS
# redirector @@redirector_url@@
def run(ctx):
    pass
/


create external SCALAR SCRIPT
bottleneck_varchar10(i VARCHAR(20))
RETURNS VARCHAR(10) AS
# redirector @@redirector_url@@

def run(ctx):
    return ctx.i
/

create external SCALAR SCRIPT
bottleneck_char10(i VARCHAR(20))
RETURNS CHAR(10) AS
# redirector @@redirector_url@@

def run(ctx):
    return ctx.i
/

create external SCALAR SCRIPT
bottleneck_decimal5(i DECIMAL(20, 0))
RETURNS DECIMAL(5, 0) AS
# redirector @@redirector_url@@

def run(ctx):
    return ctx.i
/

-- vim: ts=4:sts=4:sw=4:et:fdm=indent

