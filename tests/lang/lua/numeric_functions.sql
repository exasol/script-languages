CREATE lua SCALAR SCRIPT
pi()
RETURNS double AS

function run(ctx)
	return math.pi
end
/

CREATE lua SCALAR SCRIPT
double_mult("x" double, "y" double)
RETURNS double AS

function run(context)
	if context.x ~= null and context.y ~= null then
		return context.x * context.y
	end
end
/

CREATE lua SCALAR SCRIPT add_two_doubles(x DOUBLE, y DOUBLE) RETURNS DOUBLE AS
function run(ctx)
    if ctx.x ~= null and ctx.y ~= null then
        return ctx.x + ctx.y;
    end
    return null
end
/

CREATE lua SCALAR SCRIPT
add_three_doubles(x DOUBLE, y DOUBLE, z DOUBLE)
RETURNS DOUBLE AS
function run(ctx)
    if ctx.x ~= null and ctx.y ~= null and ctx.z ~= null then
        return ctx.x + ctx.y + ctx.z;
    end
    return null
end
/

CREATE lua SCALAR SCRIPT
split_integer_into_digits("x" DOUBLE)
EMITS (y DOUBLE) AS
function run(ctx)
    if ctx.x ~= null then
        y = math.abs(ctx.x)
        while y > 0 do
            a = y % 10
            y = y - a
            y = y / 10
            ctx.emit(a)
        end
    end
end
/

-- vim: ts=2:sts=2:sw=2
