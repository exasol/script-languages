CREATE JAVA SET SCRIPT VAREMIT_SIMPLE_SET (a double) EMITS (...) AS
class VAREMIT_SIMPLE_SET {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1);
  }
}
/
CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_SIMPLE_SET (a double) EMITS (...) AS
class DEFAULT_VAREMIT_SIMPLE_SET {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1);
  }
  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
    return "x double";
  }
}
/
--
CREATE JAVA SCALAR SCRIPT VAREMIT_SIMPLE_SCALAR (a double) EMITS (...) AS
class VAREMIT_SIMPLE_SCALAR {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1);
  }
}
/
CREATE JAVA SCALAR SCRIPT DEFAULT_VAREMIT_SIMPLE_SCALAR (a double) EMITS (...) AS
class DEFAULT_VAREMIT_SIMPLE_SCALAR {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1);
  }
  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
    return "x double";
  }
}
/
--
CREATE JAVA SCALAR SCRIPT VAREMIT_SIMPLE_ALL_DYN (...) EMITS (...) AS
class VAREMIT_SIMPLE_ALL_DYN {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1);
  }
}
/
CREATE JAVA SCALAR SCRIPT DEFAULT_VAREMIT_SIMPLE_ALL_DYN (...) EMITS (...) AS
class DEFAULT_VAREMIT_SIMPLE_ALL_DYN {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1);
  }
  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
    return "x double";
  }
}
/
--
CREATE JAVA SET SCRIPT VAREMIT_SIMPLE_SYNTAX_VAR (...) EMITS ( ...   ) AS
class VAREMIT_SIMPLE_SYNTAX_VAR {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1);
  }
}
/
CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_SIMPLE_SYNTAX_VAR (...) EMITS ( ...   ) AS
class DEFAULT_VAREMIT_SIMPLE_SYNTAX_VAR {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1);
  }
  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
    return "x double";
  }
}
/
--
CREATE JAVA SET SCRIPT VAREMIT_GENERIC_EMIT (a varchar(100)) EMITS (...) AS
class VAREMIT_GENERIC_EMIT {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    int cols = (int)exa.getOutputColumnCount();
    Object[] ret = new Object[cols];
    for (int i=0; i<cols; i++) {
      ret[i] = ctx.getString(0);
    }
    ctx.emit(ret);
  }
}
/
CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_GENERIC_EMIT (a varchar(100)) EMITS (...) AS
class DEFAULT_VAREMIT_GENERIC_EMIT {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    int cols = (int)exa.getOutputColumnCount();
    Object[] ret = new Object[cols];
    for (int i=0; i<cols; i++) {
      ret[i] = ctx.getString(0);
    }
    ctx.emit(ret);
  }
  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
    return "a varchar(100)";
  }
}
/
--
CREATE JAVA SET SCRIPT VAREMIT_ALL_GENERIC (...) EMITS (...) AS
class VAREMIT_ALL_GENERIC {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    int cols = (int)exa.getOutputColumnCount();
    Object[] ret = new Object[cols];
    for (int i=0; i<cols; i++) {
      ret[i] = ctx.getString(0);
    }
    ctx.emit(ret);
  }
}
/
CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_ALL_GENERIC (...) EMITS (...) AS
class DEFAULT_VAREMIT_ALL_GENERIC {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    int cols = (int)exa.getOutputColumnCount();
    Object[] ret = new Object[cols];
    for (int i=0; i<cols; i++) {
      ret[i] = ctx.getString(0);
    }
    ctx.emit(ret);
  }
  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
    return "a varchar(100)";
  }
}
/
--
CREATE JAVA SET SCRIPT VAREMIT_METADATA_SET_EMIT (...) EMITS(...) AS
class VAREMIT_METADATA_SET_EMIT {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        ctx.emit(Long.toString(exa.getOutputColumnCount()), 1);
        for (int i = 0; i < exa.getOutputColumnCount(); i++) {
            ctx.emit(exa.getOutputColumnName(i), 1);
            ctx.emit(exa.getOutputColumnType(i).getCanonicalName(), 1);
            ctx.emit(exa.getOutputColumnSqlType(i), 1);
            ctx.emit(Long.toString(exa.getOutputColumnPrecision(i)), 1);
            ctx.emit(Long.toString(exa.getOutputColumnScale(i)), 1);
            ctx.emit(Long.toString(exa.getOutputColumnLength(i)), 1);
        }
    }
}
/
CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_METADATA_SET_EMIT (...) EMITS(...) AS
class DEFAULT_VAREMIT_METADATA_SET_EMIT {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        ctx.emit(Long.toString(exa.getOutputColumnCount()), 1);
        for (int i = 0; i < exa.getOutputColumnCount(); i++) {
            ctx.emit(exa.getOutputColumnName(i), 1);
            ctx.emit(exa.getOutputColumnType(i).getCanonicalName(), 1);
            ctx.emit(exa.getOutputColumnSqlType(i), 1);
            ctx.emit(Long.toString(exa.getOutputColumnPrecision(i)), 1);
            ctx.emit(Long.toString(exa.getOutputColumnScale(i)), 1);
            ctx.emit(Long.toString(exa.getOutputColumnLength(i)), 1);
        }
    }
  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
    return "a varchar(123), b double";
  }
}
/
--
CREATE JAVA SET SCRIPT VAREMIT_NON_VAR_EMIT (...) EMITS (a double) AS
class VAREMIT_NON_VAR_EMIT {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1);
  }
}
/
CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_NON_VAR_EMIT (...) EMITS (a double) AS
class DEFAULT_VAREMIT_NON_VAR_EMIT {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1);
  }
  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
    return "a int";
  }
}
/
--
CREATE JAVA SET SCRIPT VAREMIT_SIMPLE_RETURNS (a int) RETURNS int AS
class VAREMIT_SIMPLE_RETURNS {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1);
  }
}
/
CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_SIMPLE_RETURNS (a int) RETURNS int AS
class DEFAULT_VAREMIT_SIMPLE_RETURNS {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1);
  }
  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
    return "x double";
  }
}
/
--
CREATE JAVA SET SCRIPT VAREMIT_EMIT_INPUT (...) EMITS (...) AS
class VAREMIT_EMIT_INPUT {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    int cols = (int)exa.getInputColumnCount();
    Object[] ret = new Object[cols];
    for (int i=0; i<cols; i++) {
      if (exa.getInputColumnType(i) == Integer.class) {
        ret[i] = ctx.getInteger(i);
      } else if (exa.getInputColumnType(i) == String.class) {
        ret[i] = ctx.getString(i);
      } else if (exa.getInputColumnType(i) == Double.class) {
        ret[i] = ctx.getDouble(i);
      } else if (exa.getInputColumnType(i) == Long.class) {
        ret[i] = ctx.getLong(i);
      } else {
        throw new RuntimeException("type not supported: " + exa.getInputColumnType(i));
      }
    }
    ctx.emit(ret);
  }
}
/
CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_EMIT_INPUT (...) EMITS (...) AS
class DEFAULT_VAREMIT_EMIT_INPUT {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    int cols = (int)exa.getInputColumnCount();
    Object[] ret = new Object[cols];
    for (int i=0; i<cols; i++) {
      if (exa.getInputColumnType(i) == Integer.class) {
        ret[i] = ctx.getInteger(i);
      } else if (exa.getInputColumnType(i) == String.class) {
        ret[i] = ctx.getString(i);
      } else if (exa.getInputColumnType(i) == Double.class) {
        ret[i] = ctx.getDouble(i);
      } else if (exa.getInputColumnType(i) == Long.class) {
        ret[i] = ctx.getLong(i);
      } else {
        throw new RuntimeException("type not supported: " + exa.getInputColumnType(i));
      }
    }
    ctx.emit(ret);
  }
  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
    return "a int";
  }
}
/
--
CREATE JAVA SET SCRIPT VAREMIT_EMIT_INPUT_WITH_META_CHECK (...) EMITS (...) AS
class VAREMIT_EMIT_INPUT_WITH_META_CHECK {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    int cols = (int)exa.getInputColumnCount();
    Object[] ret = new Object[cols];
    for (int i=0; i<cols; i++) {
      if (exa.getInputColumnType(i) == Integer.class) {
        ret[i] = ctx.getInteger(i);
      } else if (exa.getInputColumnType(i) == String.class) {
        ret[i] = ctx.getString(i);
      } else if (exa.getInputColumnType(i) == Double.class) {
        ret[i] = ctx.getDouble(i);
      } else if (exa.getInputColumnType(i) == Long.class) {
        ret[i] = ctx.getLong(i);
      } else {
        throw new RuntimeException("type not supported: " + exa.getInputColumnType(i));
      }
      assert exa.getOutputColumnType(i) == exa.getInputColumnType(i);
    }
    ctx.emit(ret);
  }
}
/
CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_EMIT_INPUT_WITH_META_CHECK (...) EMITS (...) AS
class DEFAULT_VAREMIT_EMIT_INPUT_WITH_META_CHECK {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    int cols = (int)exa.getInputColumnCount();
    Object[] ret = new Object[cols];
    for (int i=0; i<cols; i++) {
      if (exa.getInputColumnType(i) == Integer.class) {
        ret[i] = ctx.getInteger(i);
      } else if (exa.getInputColumnType(i) == String.class) {
        ret[i] = ctx.getString(i);
      } else if (exa.getInputColumnType(i) == Double.class) {
        ret[i] = ctx.getDouble(i);
      } else if (exa.getInputColumnType(i) == Long.class) {
        ret[i] = ctx.getLong(i);
      } else {
        throw new RuntimeException("type not supported: " + exa.getInputColumnType(i));
      }
      assert exa.getOutputColumnType(i) == exa.getInputColumnType(i);
    }
    ctx.emit(ret);
  }
  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
    return "a varchar(123), b double";
  }
}
/



CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_EMPTY_DEF (A DOUBLE) EMITS (...) AS
class DEFAULT_VAREMIT_EMPTY_DEF {
  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
    ctx.emit(1.4);
  }
  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
    return "";
  }
}
/

