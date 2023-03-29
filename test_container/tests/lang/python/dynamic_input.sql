CREATE PYTHON SCALAR SCRIPT
metadata_scalar_emit (...)
EMITS("v" VARCHAR(2000)) AS
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

CREATE PYTHON SCALAR SCRIPT
metadata_scalar_return (...)
RETURNS VARCHAR(2000) AS
def run(ctx):
    return repr(exa.meta.input_column_count)
/

CREATE PYTHON SCALAR SCRIPT
basic_scalar_emit( ... )
EMITS ("v" VARCHAR(2000)) as
def run(ctx):
    i = 0
    while i < exa.meta.input_column_count:
        ctx.emit(repr(ctx[i]))
        i = i + 1
/

CREATE PYTHON SCALAR SCRIPT
basic_scalar_return( ... )
RETURNS VARCHAR(2000) as
def run(ctx):
    return repr(ctx[exa.meta.input_column_count-1])
/

CREATE PYTHON SET SCRIPT
basic_set_emit( ... )
EMITS ("v" VARCHAR(2000)) as
def run(ctx):
        var = 'result: '
        while True:
                for i in range (0,exa.meta.input_column_count):
                        ctx.emit(repr(ctx[i]))
                        var = var + repr(ctx[i]) + ' , '
                if not ctx.next(): break
        ctx.emit(var)
/

CREATE PYTHON SET SCRIPT
basic_set_return( ... )
RETURNS VARCHAR(2000) as
def run(ctx):
        var = 'result: '
        while True:
                for i in range (0,exa.meta.input_column_count):
                        var = var + repr(ctx[i]) + ' , '
                if not ctx.next(): break
        return var
/

CREATE PYTHON SET SCRIPT
type_specific_add(...)
RETURNS VARCHAR(2000) as
def run(ctx):
        var = 'result: '
        if repr(exa.meta.input_columns[0].type) == "<type 'unicode'>" or repr(exa.meta.input_columns[0].type) == "<class 'str'>":
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

CREATE PYTHON SCALAR SCRIPT
wrong_arg(...)
returns varchar(2000) as
def run(ctx):
    return ctx[1]
/

CREATE PYTHON SCALAR SCRIPT
wrong_operation(...)
returns varchar(2000) as
def run(ctx):
    return ctx[0] * ctx[1]
/

CREATE PYTHON SET SCRIPT
empty_set_returns( ... )
returns varchar(2000) as
def run(ctx):
    return 1
/

CREATE PYTHON SET SCRIPT
empty_set_emits( ... )
emits (x varchar(2000)) as
def run(ctx):
    return 1
/
