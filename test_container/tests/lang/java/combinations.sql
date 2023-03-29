create java set script
SET_RETURNS(x double, y double)
returns double as
class SET_RETURNS {
    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        double acc = 0.0;
        while (true) {
            acc = acc + ctx.getDouble("x") + ctx.getDouble("y");
            if (!ctx.next())
                break;
        }
        return acc;
    }
}
/

create java set script
SET_EMITS(x double, y double)
emits (x double, y double) as
class SET_EMITS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        while (true) {
            ctx.emit(ctx.getDouble("y"), ctx.getDouble("x"));
            if (!ctx.next())
                break;
        }
    }
}
/

create java scalar script
SCALAR_RETURNS(x double, y double)
returns double as
class SCALAR_RETURNS {
    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return ctx.getDouble("x") + ctx.getDouble("y");
    }
}
/

create java scalar script
SCALAR_EMITS(x double, y double)
emits (x double, y double) as
class SCALAR_EMITS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        for (int i = ctx.getDouble("x").intValue(); i < (int)(ctx.getDouble("y") + 1); i++)
            ctx.emit((float) i, (float) (i * i));
    }
}
/
