create external SCALAR SCRIPT
metadata_scalar_emit (...)
EMITS("v" VARCHAR(2000)) AS
# redirector @@redirector_url@@
def run(ctx):
    ctx.emit(repr(exa.meta.input_column_count))
    for i in range (0,exa.meta.input_column_count):
        ctx.emit(exa.meta.input_columns[i].name)
        ctx.emit(repr(exa.meta.input_columns[i].type))
        ctx.emit(exa.meta.input_columns[i].sql_type)
        ctx.emit(repr(exa.meta.input_columns[i].precision))
        ctx.emit(repr(exa.meta.input_columns[i].scale))
        ctx.emit(repr(exa.meta.input_columns[i].length))
/

create external SCALAR SCRIPT
metadata_scalar_return (...)
RETURNS VARCHAR(2000) AS
# redirector @@redirector_url@@
def run(ctx):
    return repr(exa.meta.input_column_count)
/

create external SCALAR SCRIPT
basic_scalar_emit( ... )
EMITS ("v" VARCHAR(2000)) AS
# redirector @@redirector_url@@
def run(ctx):
    i = 0
    while i < exa.meta.input_column_count:
        ctx.emit(repr(ctx[i]))
        i = i + 1
/

create external SCALAR SCRIPT
basic_scalar_return( ... )
RETURNS VARCHAR(2000) AS
# redirector @@redirector_url@@
def run(ctx):
    return repr(ctx[exa.meta.input_column_count-1])
/

create external SET SCRIPT
basic_set_emit( ... )
EMITS ("v" VARCHAR(2000)) AS
# redirector @@redirector_url@@
def run(ctx):
        var = 'result: '
        while True:
                for i in range (0,exa.meta.input_column_count):
                        ctx.emit(repr(ctx[i]))
                        var = var + repr(ctx[i]) + ' , '
                if not ctx.next(): break
        ctx.emit(var)
/

create external SET SCRIPT
basic_set_return( ... )
RETURNS VARCHAR(2000) AS
# redirector @@redirector_url@@
def run(ctx):
        var = 'result: '
        while True:
                for i in range (0,exa.meta.input_column_count):
                        var = var + repr(ctx[i]) + ' , '
                if not ctx.next(): break
        return var
/

create external SET SCRIPT
type_specific_add(...)
RETURNS VARCHAR(2000) AS
# redirector @@redirector_url@@
def run(ctx):
        var = 'result: '
        if repr(exa.meta.input_columns[0].type) == "<type 'unicode'>":
                while True:
                        for i in range (0,exa.meta.input_column_count):
                                var = var + ctx[i] + ' , '
                        if not ctx.next(): break
        else:
                sum = 0
                while True:
                        for i in range (0,exa.meta.input_column_count):
                                sum = sum + ctx[i]
                        if not ctx.next(): break
                var = var + repr(sum)
        return var
/

create external SCALAR SCRIPT
wrong_arg(...)
returns varchar(2000) AS
# redirector @@redirector_url@@
def run(ctx):
    return ctx[1]
/

create external SCALAR SCRIPT
wrong_operation(...)
returns varchar(2000) AS
# redirector @@redirector_url@@
def run(ctx):
    return ctx[0] * ctx[1]
/

create external SET SCRIPT
empty_set_returns( ... )
returns varchar(2000) AS
# redirector @@redirector_url@@
def run(ctx):
    return 1
/

create external SET SCRIPT
empty_set_emits( ... )
emits (x varchar(2000)) AS
# redirector @@redirector_url@@
def run(ctx):
    return 1
/
