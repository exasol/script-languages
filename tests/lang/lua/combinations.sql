create lua set script
SET_RETURNS(x double, y double)
returns double as
function run(ctx)
local acc = 0.0
repeat
acc = acc + ctx.x + ctx.y
until not ctx.next()
        return acc
end
/



create lua set script
SET_EMITS(x double, y double)
emits (x double, y double) as
function run(ctx)
repeat
                ctx.emit(ctx.y, ctx.x)
                until not ctx.next()
end
/



create lua scalar script
SCALAR_RETURNS(x double, y double)
returns double as
function run(ctx)
     return ctx.x + ctx.y
end
/


create lua scalar script
SCALAR_EMITS(x double, y double)
emits (x double, y double) as
function run(ctx)
         for i = ctx.x, ctx.y do
             ctx.emit(i, i * i)
         end
end
/
