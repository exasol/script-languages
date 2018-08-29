CREATE LUA SCALAR SCRIPT
metadata_scalar_emit (...)
EMITS("v" VARCHAR(2000)) AS
function run(data)
        data.emit(exa.meta.input_column_count)
    for i=1,exa.meta.input_column_count do
        data.emit(exa.meta.input_columns[i].name)
        data.emit(exa.meta.input_columns[i].type)
        data.emit(exa.meta.input_columns[i].sql_type)
        data.emit(exa.meta.input_columns[i].precision)
        data.emit(exa.meta.input_columns[i].scale)
        data.emit(exa.meta.input_columns[i].length)
    end
end
/

CREATE LUA SCALAR SCRIPT
metadata_scalar_return (...)
RETURNS VARCHAR(2000) AS
function run(ctx)
        return exa.meta.input_column_count
end
/

CREATE LUA SCALAR SCRIPT
basic_scalar_emit( ... )
EMITS ("v" VARCHAR(2000)) as
function run(ctx)
    for i=1,exa.meta.input_column_count do
        ctx.emit( ctx[i] )
    end
end
/

CREATE LUA SCALAR SCRIPT
basic_scalar_return( ... )
RETURNS VARCHAR(2000) as
function run(ctx)
    return ctx[exa.meta.input_column_count]
end
/

CREATE LUA SET SCRIPT
basic_set_emit( ... )
EMITS ("v" VARCHAR(2000)) as
function run(ctx)
    var = 'result: '
    repeat
        for i=1,exa.meta.input_column_count do
            ctx.emit(ctx[i])
            var = var .. ctx[i] .. ' , '
        end
    until not ctx.next()
    ctx.emit(var)
end
/

CREATE LUA SET SCRIPT
basic_set_return( ... )
RETURNS VARCHAR(2000) as
function run(ctx)
    var = 'result: '
    repeat
        for i=1,exa.meta.input_column_count do
            var = var .. ctx[i] .. ' , '
        end
    until not ctx.next()
    return var
end
/

CREATE LUA SET SCRIPT
type_specific_add(...)
RETURNS VARCHAR(2000) as
function run(ctx)
        var = 'result: '
        -- all parameters have to have the same type
        if exa.meta.input_columns[1].type == 'string' then
                repeat
                for i=1,exa.meta.input_column_count do
                        var = var .. ctx[i] .. ' , '
                end
                until not ctx.next()
        else
                sum = 0
                repeat
                for i=1,exa.meta.input_column_count do
                        sum = sum + ctx[i]
                end
                until not ctx.next()
                var = var .. sum
        end
        return var
end
/

CREATE LUA SCALAR SCRIPT
wrong_arg(...)
returns varchar(2000) as
        function run(ctx)
                return ctx[2]
        end
/

CREATE LUA SCALAR SCRIPT
wrong_operation(...)
returns varchar(2000) as
        function run(ctx)
                return ctx[1] + ctx[2]
        end
/

CREATE LUA SET SCRIPT
empty_set_returns( ... )
returns varchar(2000) as
function run(ctx)
        return 1
end
/

CREATE LUA SET SCRIPT
empty_set_emits( ... )
emits (x varchar(2000)) as
function run(ctx)
        return 1
end
/

