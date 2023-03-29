CREATE JAVA SCALAR SCRIPT
metadata_scalar_emit (...)
EMITS("v" VARCHAR(2000)) AS
class METADATA_SCALAR_EMIT {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        ctx.emit(Long.toString(exa.getInputColumnCount()));
        for (int i = 0; i < exa.getInputColumnCount(); i++) {
            ctx.emit(exa.getInputColumnName(i));
            ctx.emit(exa.getInputColumnType(i).getCanonicalName());
            ctx.emit(exa.getInputColumnSqlType(i));
            ctx.emit(Long.toString(exa.getInputColumnPrecision(i)));
            ctx.emit(Long.toString(exa.getInputColumnScale(i)));
            ctx.emit(Long.toString(exa.getInputColumnLength(i)));
        }
    }
}
/

CREATE JAVA SCALAR SCRIPT
metadata_scalar_return (...)
RETURNS VARCHAR(2000) AS
class METADATA_SCALAR_RETURN {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return Long.toString(exa.getInputColumnCount());
    }
}
/

CREATE JAVA SCALAR SCRIPT
basic_scalar_emit( ... )
EMITS ("v" VARCHAR(2000)) as
import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Timestamp;
class BASIC_SCALAR_EMIT {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        int i = 0;
        while (i < exa.getInputColumnCount()) {
            Class cls = exa.getInputColumnType(i);
            if (cls == Integer.class) {
                ctx.emit(ctx.getInteger(i).toString());
            }
            else if (cls == Long.class) {
                ctx.emit(ctx.getLong(i).toString());
            }
            else if (cls == Class.forName("java.math.BigDecimal")) {
                ctx.emit(ctx.getBigDecimal(i).toString());
            }
            else if (cls == Double.class) {
                ctx.emit(ctx.getDouble(i).toString());
            }
            else if (cls == String.class) {
                ctx.emit(ctx.getString(i));
            }
            else if (cls == Boolean.class) {
                ctx.emit(ctx.getBoolean(i).toString());
            }
            else if (cls == Class.forName("java.sql.Date")) {
                ctx.emit(ctx.getDate(i).toString());
            }
            else if (cls == Class.forName("java.sql.Timestamp")) {
                ctx.emit(ctx.getTimestamp(i).toString());
            }
            i = i + 1;
        }
    }
}
/

CREATE JAVA SCALAR SCRIPT
basic_scalar_return( ... )
RETURNS VARCHAR(2000) as
class BASIC_SCALAR_RETURN {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        int col = (int) exa.getInputColumnCount() - 1;
        Class cls = exa.getInputColumnType(col);
        if (cls == Integer.class) {
            return ctx.getInteger(col).toString();
        }
        else if (cls == Long.class) {
            return ctx.getLong(col).toString();
        }
        else if (cls == Class.forName("java.math.BigDecimal")) {
            return ctx.getBigDecimal(col).toString();
        }
        else if (cls == Double.class) {
            return ctx.getDouble(col).toString();
        }
        else if (cls == String.class) {
            return ctx.getString(col);
        }
        else if (cls == Boolean.class) {
            return ctx.getBoolean(col).toString();
        }
        else if (cls == Class.forName("java.sql.Date")) {
            return ctx.getDate(col).toString();
        }
        else if (cls == Class.forName("java.sql.Timestamp")) {
            return ctx.getTimestamp(col).toString();
        }
        else {
            return null;
        }
    }
}
/

CREATE JAVA SET SCRIPT
basic_set_emit( ... )
EMITS ("v" VARCHAR(2000)) as
import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Timestamp;
class BASIC_SET_EMIT {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        String var = "result: ";
        while (true) {
            for (int i = 0; i < exa.getInputColumnCount(); i++) {
                Class cls = exa.getInputColumnType(i);
                if (cls == Integer.class) {
                    ctx.emit(ctx.getInteger(i).toString());
                    var += ctx.getInteger(i).toString() + " , ";
                }
                else if (cls == Long.class) {
                    ctx.emit(ctx.getLong(i).toString());
                    var += ctx.getLong(i).toString() + " , ";
                }
                else if (cls == Class.forName("java.math.BigDecimal")) {
                    ctx.emit(ctx.getBigDecimal(i).toString());
                    var += ctx.getBigDecimal(i).toString() + " , ";
                }
                else if (cls == Double.class) {
                    ctx.emit(ctx.getDouble(i).toString());
                    var += ctx.getDouble(i).toString() + " , ";
                }
                else if (cls == String.class) {
                    ctx.emit(ctx.getString(i));
                    var += ctx.getString(i) + " , ";
                }
                else if (cls == Boolean.class) {
                    ctx.emit(ctx.getBoolean(i).toString());
                    var += ctx.getBoolean(i).toString() + " , ";
                }
                else if (cls == Class.forName("java.sql.Date")) {
                    ctx.emit(ctx.getDate(i).toString());
                    var += ctx.getDate(i).toString() + " , ";
                }
                else if (cls == Class.forName("java.sql.Timestamp")) {
                    ctx.emit(ctx.getTimestamp(i).toString());
                    var += ctx.getTimestamp(i).toString() + " , ";
                }
            }
            if (!ctx.next())
                break;
        }
        ctx.emit(var);
    }
}
/

CREATE JAVA SET SCRIPT
basic_set_return( ... )
RETURNS VARCHAR(2000) as
import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Timestamp;
class BASIC_SET_RETURN {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        String var = "result: ";
        while (true) {
            for (int i = 0; i < exa.getInputColumnCount(); i++) {
                Class cls = exa.getInputColumnType(i);
                if (cls == Integer.class) {
                    var += ctx.getInteger(i).toString() + " , ";
                }
                else if (cls == Long.class) {
                    var += ctx.getLong(i).toString() + " , ";
                }
                else if (cls == Class.forName("java.math.BigDecimal")) {
                    var += ctx.getBigDecimal(i).toString() + " , ";
                }
                else if (cls == Double.class) {
                    var += ctx.getDouble(i).toString() + " , ";
                }
                else if (cls == String.class) {
                    var += ctx.getString(i) + " , ";
                }
                else if (cls == Boolean.class) {
                    var += ctx.getBoolean(i).toString() + " , ";
                }
                else if (cls == Class.forName("java.sql.Date")) {
                    var += ctx.getDate(i).toString() + " , ";
                }
                else if (cls == Class.forName("java.sql.Timestamp")) {
                    var += ctx.getTimestamp(i).toString() + " , ";
                }
            }
            if (!ctx.next())
                break;
        }
        return var;
    }
}
/

CREATE JAVA SET SCRIPT
type_specific_add(...)
RETURNS VARCHAR(2000) as
import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Timestamp;
class TYPE_SPECIFIC_ADD {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        String var = "result: ";
        Class cls = exa.getInputColumnType(0);
        if (cls == String.class) {
            while (true) {
                for (int i = 0; i < exa.getInputColumnCount(); i++) {
                    var += ctx.getString(i) + " , ";
                }
                if (!ctx.next())
                    break;
            }
        }
        else if (cls == Integer.class || cls == Long.class || cls == Double.class) {
            double sum = 0;
            while (true) {
                for (int i = 0; i < exa.getInputColumnCount(); i++) {
                    sum += ctx.getDouble(i);
                }
                if (!ctx.next())
                    break;
            }
            var += Double.toString(sum);
        }
        return var;
    }
}
/

CREATE JAVA SCALAR SCRIPT
wrong_arg(...)
returns varchar(2000) as
class WRONG_ARG {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return ctx.getString(1);
    }
}
/

CREATE JAVA SCALAR SCRIPT
wrong_operation(...)
returns varchar(2000) as
class WRONG_OPERATION {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return ctx.getString(0) * ctx.getString(1);
    }
}
/

CREATE JAVA SET SCRIPT
empty_set_returns( ... )
returns varchar(2000) as
class EMPTY_SET_RETURNS {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return Integer.toString(1);
    }
}
/

CREATE JAVA SET SCRIPT
empty_set_emits( ... )
emits (x varchar(2000)) as
class EMPTY_SET_EMITS {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        emit(Integer.toString(1));
    }
}
/
