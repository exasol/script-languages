create lua scalar script dob_1i_1o(x double) emits(y double)
as
    function run(ctx)
        ctx.emit(ctx.x)
        ctx.emit(ctx.x)
    end
/
create lua scalar script line_1i_1o(x double) emits(y double)
as
    function run(ctx)
        ctx.emit(ctx.x)
    end
/
create lua scalar script line_1i_2o(x double) emits(y double, z double)
as
    function run(ctx)
        ctx.emit(ctx.x, ctx.x)
    end
/
create lua scalar script line_2i_1o(x double, y double) emits(z double)
as
    function run(ctx)
        ctx.emit(ctx.x + ctx.y)
    end
/
create lua scalar script line_3i_2o(x double, y double, z double) emits(z1 double, z2 double)
as
    function run(ctx)
        ctx.emit(ctx.x + ctx.y, 3000)
    end
/
