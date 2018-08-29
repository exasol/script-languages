CREATE lua SCALAR SCRIPT
basic_emit_several_groups(a DOUBLE, b DOUBLE)
EMITS (i DOUBLE, j VARCHAR(40)) AS
function run(ctx)
    for n = 0, ctx.a, 1 do
        for i = 0, ctx.b, 1 do
            ctx.emit(i, exa.meta.vm_id)
        end
    end
end
/

CREATE lua SET SCRIPT
basic_test_reset(i DOUBLE, j VARCHAR(40))
EMITS (k DOUBLE) AS
function run(ctx)
    ctx.emit(ctx.i)
    ctx.next()
    ctx.emit(ctx.i)
    ctx.reset()
    ctx.emit(ctx.i)
    ctx.next()
    ctx.emit(ctx.i)
end
/

CREATE lua SCALAR SCRIPT
basic_pi()
RETURNS DOUBLE AS

function run(ctx)
	return math.pi
end
/

CREATE lua SCALAR SCRIPT
basic_e()
EMITS (pow DOUBLE) AS

function run(ctx)
	for i=0, 10 do
		ctx.emit(math.exp(i))
	end
end
/

CREATE lua SCALAR SCRIPT
basic_mult("x" DOUBLE, "y" DOUBLE)
RETURNS DOUBLE AS

function run(ctx)
	if ctx.x ~= null and ctx.y ~= null then
		return ctx.x * ctx.y
	else
		return null
	end
end
/

CREATE lua SET SCRIPT
"WordCase"("Words" VARCHAR(100))
EMITS ("lowercase" DOUBLE, "UPPERCASE" DOUBLE, "CamelCase" DOUBLE) AS

function run(ctx)
	local l = 0
	local u = 0
	local c = 0
	for word in string.gmatch(ctx.Words, "%w+") do
		if word == string.lower(word) then
			l = l + 1
		elseif word == string.upper(word) then
			u = u + 1
		else
			c = c + 1
		end
	end
	ctx.emits(l, u, c)
end
/

CREATE lua SCALAR SCRIPT
basic_emit_two_ints()
EMITS (i DOUBLE, j DOUBLE) AS
function run(ctx)
	ctx.emit(1,2)
end
/


CREATE lua SCALAR SCRIPT
basic_nth_partial_sum(n double)
RETURNS double as

function run(ctx)
    if ctx.n ~= null then
        return ctx.n * (ctx.n + 1) / 2
    else
        return 0
    end
end
/


CREATE lua SCALAR SCRIPT
basic_range(n double)
EMITS (n double) AS

function run(ctx)
    if ctx.n ~= null then
        local i = 0
        while i < ctx.n do
            ctx.emit(i)
            i = i + 1
        end
    end
end
/

CREATE lua SET SCRIPT
basic_sum(x double)
RETURNS double AS

function run(ctx)
    local s = 0
    repeat
        if ctx.x ~= null then
            s = s + ctx.x
        end
    until not ctx.next()
    return s
end
/

CREATE lua SET SCRIPT
basic_sum_grp(x double)
EMITS (s double) AS

function run(ctx)
    local s = 0
    repeat
        if ctx.x ~= null then
            s = s + ctx.x
        end
    until not ctx.next()
    ctx.emit(s)
end
/

CREATE lua SET SCRIPT
set_returns_has_empty_input(a double) RETURNS boolean AS
function run(ctx)
    if ctx.x == null then
        return true
    else
        return false
    end
end                                                                          
/

CREATE lua SET SCRIPT
set_emits_has_empty_input(a double) EMITS (x double, y varchar(10)) AS
function run(ctx)
    if ctx.x == null then
        ctx.emit(1,'1')
    else
        ctx.emit(2,'2')
    end
end
/

-- vim: ts=4:sts=4:sw=4
