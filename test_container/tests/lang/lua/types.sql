CREATE lua SCALAR SCRIPT echo_boolean(x BOOLEAN) RETURNS BOOLEAN AS
function run(ctx)
    if ctx.x ~= null then
        return ctx.x
    end
    return null
end
/

CREATE lua SCALAR SCRIPT echo_char1(x CHAR(1)) RETURNS CHAR(1) AS
function run(ctx)
    if ctx.x ~= null then
        return ctx.x
    end
    return null
end
/

CREATE lua SCALAR SCRIPT echo_char10(x CHAR(10)) RETURNS CHAR(10) AS
function run(ctx)
    if ctx.x ~= null then
        if string.len(ctx.x) ~= 10 then
            return null                
        else
            return ctx.x
        end
    end
    return null
end
/


--
--[0A000] Feature not supported: Lua script input type not supported DATE
--
--CREATE lua SCALAR SCRIPT echo_date(x DATE) RETURNS DATE AS
--function run(ctx)
--    if ctx.x ~= null then
--        return ctx.x
--    end
--    return null
--end
--/

CREATE lua SCALAR SCRIPT echo_integer(x INTEGER) RETURNS INTEGER AS
function run(ctx)
    if ctx.x ~= null then
        return ctx.x
    end
    return null
end
/

CREATE lua SCALAR SCRIPT echo_double(x DOUBLE) RETURNS DOUBLE AS
function run(ctx)
    if ctx.x ~= null then
        return ctx.x
    end
    return null
end
/

CREATE lua SCALAR SCRIPT echo_decimal_36_0(x DECIMAL(36,0)) RETURNS DECIMAL(36,0) AS
function run(ctx)
    if ctx.x ~= null then
        return ctx.x
    end
    return null
end
/


CREATE lua SCALAR SCRIPT echo_decimal_36_36(x DECIMAL(36,36)) RETURNS DECIMAL(36,36) AS
function run(ctx)
    if ctx.x ~= null then
        return ctx.x
    end
    return null
end    
/

CREATE lua SCALAR SCRIPT echo_varchar10(x VARCHAR(10)) RETURNS VARCHAR(10) AS
function run(ctx)
    if ctx.x ~= null then
        return ctx.x
    end
    return null
end
/

--
--[0A000] Feature not supported: Lua script input type not supported TIMESTAMP
--
--CREATE lua SCALAR SCRIPT echo_timestamp(x TIMESTAMP) RETURNS TIMESTAMP AS
--function run(ctx)
--    if ctx.x ~= null then
--        return ctx.x
--    end
--    return null
--end
--/

CREATE lua SCALAR SCRIPT run_func_is_empty() RETURNS DOUBLE AS
function run(ctx)
end
/


CREATE lua SCALAR SCRIPT
bottleneck_varchar10(i VARCHAR(20))
RETURNS VARCHAR(10) AS

function run(ctx)
    return ctx.i
end
/

CREATE lua SCALAR SCRIPT
bottleneck_char10(i VARCHAR(20))
RETURNS CHAR(10) AS

function run(ctx)
    return ctx.i
end
/


-- vim: ts=4:sts=4:sw=4:et:fdm=indent
