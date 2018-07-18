--CREATE <lang>  SCALAR SCRIPT
--base_pi()
--RETURNS DOUBLE AS

-- pi

CREATE java SCALAR SCRIPT
basic_emit_several_groups(a INTEGER, b INTEGER)
EMITS (i INTEGER, j VARCHAR(40)) AS
class BASIC_EMIT_SEVERAL_GROUPS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        for (int n = 0; n < ctx.getInteger("a"); n++)
            for (int i = 0; i < ctx.getInteger("b"); i++)
                ctx.emit(i, exa.getVmId());
    }
}
/

CREATE java SET SCRIPT
basic_test_reset(i INTEGER, j VARCHAR(40))
EMITS (k INTEGER) AS
class BASIC_TEST_RESET {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        ctx.emit(ctx.getInteger("i"));
        ctx.next();
        ctx.emit(ctx.getInteger("i"));
        ctx.reset();
        ctx.emit(ctx.getInteger("i"));
        ctx.next();
        ctx.emit(ctx.getInteger("i"));
    }
}
/

CREATE java SCALAR SCRIPT
basic_emit_two_ints()
EMITS (i INTEGER, j INTEGER) AS
class BASIC_EMIT_TWO_INTS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        ctx.emit(1,2);
    }
}
/

CREATE java SCALAR SCRIPT
basic_nth_partial_sum(n INTEGER)
RETURNS INTEGER as
class BASIC_NTH_PARTIAL_SUM {
    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getInteger("n") != null)
            return ctx.getInteger("n") * (ctx.getInteger("n") + 1) / 2;
        return 0;
    }
}
/

CREATE java SCALAR SCRIPT
basic_range(n INTEGER)
EMITS (n INTEGER) AS
class BASIC_RANGE {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getInteger("n") != null)
            for (int i = 0; i < ctx.getInteger("n"); i++)
                ctx.emit(i);
    }
}
/

CREATE java SET SCRIPT
basic_sum(x INTEGER)
RETURNS INTEGER AS
class BASIC_SUM {
    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        int s = 0;
        while (true) {
            if (ctx.getInteger("x") != null)
                s += ctx.getInteger("x");
            if (!ctx.next())
                break;
        }
        return s;
    }
}
/

CREATE java SET SCRIPT
basic_sum_grp(x INTEGER)
EMITS (s INTEGER) AS
class BASIC_SUM_GRP {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        int s = 0;
        while (true) {
            if (ctx.getInteger("x") != null)
                s += ctx.getInteger("x");
            if (!ctx.next())
                break;
        }
        ctx.emit(s);
    }
}
/

CREATE java SET SCRIPT
set_returns_has_empty_input(a double) RETURNS boolean AS
class SET_RETURNS_HAS_EMPTY_INPUT {
    static boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return ctx.getInteger("x") == null;
    }
}
/

CREATE java SET SCRIPT
set_emits_has_empty_input(a double) EMITS (x double, y varchar(10)) AS
class SET_EMITS_HAS_EMPTY_INPUT {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getDouble("x") == null)
            ctx.emit(1,"1")
        else
            ctx.emit(2,"2")
    }
}
/

-- vim: ts=4:sts=4:sw=4
