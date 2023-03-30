CREATE java SCALAR SCRIPT echo_boolean(x BOOLEAN) RETURNS BOOLEAN AS
class ECHO_BOOLEAN {
    static Boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getBoolean("x") != null)
            return ctx.getBoolean("x");
        return null;
    }
}
/

CREATE java SCALAR SCRIPT echo_char1(x CHAR(1)) RETURNS CHAR(1) AS
class ECHO_CHAR1 {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getString("x") != null)
            return ctx.getString("x");
        return null;
    }
}
/

CREATE java SCALAR SCRIPT echo_char10(x CHAR(10)) RETURNS CHAR(10) AS
class ECHO_CHAR10 {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getString("x") != null && ctx.getString("x").length() == 10)
            return ctx.getString("x");
        return null;
    }
}
/

CREATE java SCALAR SCRIPT echo_date(x DATE) RETURNS DATE AS
import java.sql.Date;
class ECHO_DATE {
    static Date run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getDate("x") != null)
            return ctx.getDate("x");
        return null;
    }
}
/

CREATE java SCALAR SCRIPT echo_integer(x INTEGER) RETURNS INTEGER AS
class ECHO_INTEGER {
    static Long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getLong("x") != null)
            return ctx.getLong("x");
        return null;
    }
}
/

CREATE java SCALAR SCRIPT echo_double(x DOUBLE) RETURNS DOUBLE AS
class ECHO_DOUBLE {
    static Double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getDouble("x") != null)
            return ctx.getDouble("x");
        return null;
    }
}
/

CREATE java SCALAR SCRIPT echo_decimal_36_0(x DECIMAL(36,0)) RETURNS DECIMAL(36,0) AS
import java.math.BigDecimal;
class ECHO_DECIMAL_36_0 {
    static BigDecimal run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getBigDecimal("x") != null)
            return ctx.getBigDecimal("x");
        return null;
    }
}
/

CREATE java SCALAR SCRIPT echo_decimal_36_36(x DECIMAL(36,36)) RETURNS DECIMAL(36,36) AS
import java.math.BigDecimal;
class ECHO_DECIMAL_36_36 {
    static BigDecimal run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getBigDecimal("x") != null)
            return ctx.getBigDecimal("x");
        return null;
    }
}
/


CREATE java SCALAR SCRIPT echo_varchar10(x VARCHAR(10)) RETURNS VARCHAR(10) AS
class ECHO_VARCHAR10 {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getString("x") != null)
            return ctx.getString("x");
        return null;
    }
}
/

CREATE java SCALAR SCRIPT echo_timestamp(x TIMESTAMP) RETURNS TIMESTAMP AS
import java.sql.Timestamp;
class ECHO_TIMESTAMP {
    static Timestamp run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        if (ctx.getTimestamp("x") != null)
            return ctx.getTimestamp("x");
        return null;
    }
}
/

CREATE java SCALAR SCRIPT run_func_is_empty() RETURNS DOUBLE AS
class RUN_FUNC_IS_EMPTY {
    static Double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return null;
    }
}
/


CREATE java SCALAR SCRIPT
bottleneck_varchar10(i VARCHAR(20))
RETURNS VARCHAR(10) AS
class BOTTLENECK_VARCHAR10 {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return ctx.getString("i");
    }
}
/

CREATE java SCALAR SCRIPT
bottleneck_char10(i VARCHAR(20))
RETURNS CHAR(10) AS
class BOTTLENECK_CHAR10 {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return ctx.getString("i");
    }
}
/

CREATE java SCALAR SCRIPT
bottleneck_decimal5(i DECIMAL(20, 0))
RETURNS DECIMAL(5, 0) AS
import java.math.BigDecimal;
class BOTTLENECK_DECIMAL5 {
    static BigDecimal run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return ctx.getBigDecimal("i");
    }
}
/

-- vim: ts=4:sts=4:sw=4:et:fdm=indent

