CREATE JAVA SCALAR SCRIPT
get_database_name() returns varchar(300) AS
class GET_DATABASE_NAME {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getDatabaseName();
    }
}
/
create java scalar script
get_database_version() returns varchar(20) as
class GET_DATABASE_VERSION {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getDatabaseVersion();
    }
}
/
create java scalar script
get_script_language() emits (s1 varchar(300), s2 varchar(300)) as
class GET_SCRIPT_LANGUAGE {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        ctx.emit(exa.getScriptLanguage(), "Java");
    }
}
/
create java scalar script
get_script_name() returns varchar(200) as
class GET_SCRIPT_NAME {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getScriptName();
    }
}
/
create java scalar script
get_script_schema() returns varchar(200) as
class GET_SCRIPT_SCHEMA {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getScriptSchema();
    }
}
/
create java scalar script
get_current_user() returns varchar(200) as
class GET_CURRENT_USER {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getCurrentUser();
    }
}
/
create java scalar script
get_scope_user() returns varchar(200) as
class GET_SCOPE_USER {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getScopeUser();
    }
}
/
create java scalar script
get_current_schema() returns varchar(200) as
class GET_CURRENT_SCHEMA {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getCurrentSchema();
    }
}
/
create java scalar script
get_script_code() returns varchar(2000) as
class GET_SCRIPT_CODE {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getScriptCode();
    }
}
/
create java scalar script
get_session_id() returns varchar(200) as
class GET_SESSION_ID {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getSessionId();
    }
}
/
create java scalar script
get_statement_id() returns number as
class GET_STATEMENT_ID {
    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getStatementId();
    }
}
/
create java scalar script
get_node_count() returns number as
class GET_NODE_COUNT {
    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getNodeCount();
    }
}
/
create java scalar script
get_node_id() returns number as
class GET_NODE_ID {
    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getNodeId();
    }
}
/
create java scalar script
get_vm_id() returns varchar(200) as
class GET_VM_ID {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getVmId();
    }
}
/
create java scalar script
get_input_type_scalar() returns varchar(200) as
class GET_INPUT_TYPE_SCALAR {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getInputType();
    }
}
/
create java set script
get_input_type_set(a double) returns varchar(200) as
class GET_INPUT_TYPE_SET {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getInputType();
    }
}
/
create java scalar script
get_input_column_count_scalar(c1 double, c2 varchar(100))
returns number as
class GET_INPUT_COLUMN_COUNT_SCALAR {
    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getInputColumnCount();
    }
}
/
create java set script
get_input_column_count_set(c1 double, c2 varchar(100))
returns number as
class GET_INPUT_COLUMN_COUNT_SET {
    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getInputColumnCount();
    }
}
/
create java scalar script
get_input_columns(c1 double, c2 varchar(200))
emits (column_id number, column_name varchar(200), column_type varchar(20),
 	   column_sql_type varchar(20), column_precision number, column_scale number,
  	   column_length number) as
class GET_INPUT_COLUMNS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        for (int i = 0; i < exa.getInputColumnCount(); i++) {
            String name = exa.getInputColumnName(i);
            long precision = exa.getInputColumnPrecision(i);
            String thetype = exa.getInputColumnType(i).getCanonicalName();
            String sql_type = exa.getInputColumnSqlType(i);
            long scale = exa.getInputColumnScale(i);
            long length = exa.getInputColumnLength(i);
            if (name == null)
                name = "no-name";
            if (thetype == null)
                thetype = "no-type";
            if (sql_type == null)
                sql_type = "no-sql-type";
            ctx.emit(i + 1, name, thetype, sql_type, precision, scale, length);
        }
    }
}
/
create java scalar script
get_output_type_return()
returns varchar(200) as
class GET_OUTPUT_TYPE_RETURN {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getOutputType();
    }
}
/
create java scalar script
get_output_type_emit()
emits (t varchar(200)) as
class GET_OUTPUT_TYPE_EMIT {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        ctx.emit(exa.getOutputType());
    }
}
/
create java scalar script
get_output_column_count_return()
returns number as
class GET_OUTPUT_COLUMN_COUNT_RETURN {
    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return exa.getOutputColumnCount();
    }
}
/
create java scalar script
get_output_column_count_emit()
emits (x number, y number, z number) as
class GET_OUTPUT_COLUMN_COUNT_EMIT {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        ctx.emit(exa.getOutputColumnCount(), exa.getOutputColumnCount(), exa.getOutputColumnCount());
    }
}
/
create java scalar script
get_output_columns()
emits (column_id number, column_name varchar(200), column_type varchar(20),
 	   column_sql_type varchar(20), column_precision number, column_scale number,
  	   column_length number) as
class GET_OUTPUT_COLUMNS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        for (int i = 0; i < exa.getOutputColumnCount(); i++) {
            String name = exa.getOutputColumnName(i);
            long precision = exa.getOutputColumnPrecision(i);
            String thetype = exa.getOutputColumnType(i).getCanonicalName();
            String sql_type = exa.getOutputColumnSqlType(i);
            long scale = exa.getOutputColumnScale(i);
            long length = exa.getOutputColumnLength(i);
            if (name == null)
               name = "no-name";
            if (thetype == null)
               thetype = "no-type";
            if (sql_type == null)
               sql_type = "no-sql-type";
            ctx.emit(i + 1, name, thetype, sql_type, precision, scale, length);
        }
    }
}
/

--  select get_output_columns() from dual;
-- (1, ...)

create java scalar script
get_precision_scale_length(n decimal(6,3), v varchar(10))
emits (precision1 number, scale1 number, length1 number, precision2 number, scale2 number, length2 number) as
class GET_PRECISION_SCALE_LENGTH {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        long precision1 = exa.getInputColumnPrecision(0);
        long scale1 = exa.getInputColumnScale(0);
        long length1 = exa.getInputColumnLength(0);
        long precision2 = exa.getInputColumnPrecision(1);
        long scale2 = exa.getInputColumnScale(1);
        long length2 = exa.getInputColumnLength(1);
        ctx.emit(precision1, scale1, length1, precision2, scale2, length2);
    }
}
/

create java scalar script
get_char_length(text char(10))
emits(len1 number, len2 number, dummy char(20))
as
class GET_CHAR_LENGTH {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        long v = exa.getInputColumnLength(0);
        long w = exa.getOutputColumnLength(2);
        ctx.emit(v, w, "9876543210");
    }
}
/

