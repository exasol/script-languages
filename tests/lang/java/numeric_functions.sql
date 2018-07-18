CREATE java SCALAR SCRIPT
pi()
RETURNS double AS
class PI {
    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return Math.PI;
    }
}
/

CREATE java SCALAR SCRIPT
double_mult("x" double, "y" double)
RETURNS double AS
class DOUBLE_MULT {
    static Double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getDouble("x") == null || ctx.getDouble("y") == null)
            return null;
        else
            return (double) ctx.getDouble("x") * (double) ctx.getDouble("y");
    }
}
/

CREATE java SCALAR SCRIPT add_two_doubles(x DOUBLE, y DOUBLE) RETURNS DOUBLE AS
class ADD_TWO_DOUBLES {
    static Double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getDouble("x") != null && ctx.getDouble("y") != null)
            return (double) ctx.getDouble("x") + (double) ctx.getDouble("y");
        return null;
    }
}
/

CREATE java SCALAR SCRIPT
add_three_doubles(x DOUBLE, y DOUBLE, z DOUBLE)
RETURNS DOUBLE AS
class ADD_THREE_DOUBLES {
    static Double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getDouble("x") != null && ctx.getDouble("y") != null && ctx.getDouble("z") != null)
            return (double) ctx.getDouble("x") + (double) ctx.getDouble("y") + (double) ctx.getDouble("z");
        return null;
    }
}
/

CREATE java SCALAR SCRIPT
split_integer_into_digits("x" INTEGER)
EMITS (y INTEGER) AS
class SPLIT_INTEGER_INTO_DIGITS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getInteger("x") != null) {
            int y = Math.abs(ctx.getInteger("x"));
            while (y > 0) {
                ctx.emit(y % 10);
                y /= 10;
            }
        }
    }
}
/


-- vim: ts=2:sts=2:sw=2:et
