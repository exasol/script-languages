#!/usr/bin/env python3

from exasol_python_test_framework import udf


class _JavaUdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        self.query(udf.fixindent('''
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
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_current_schema() returns varchar(200) as
            class GET_CURRENT_SCHEMA {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getCurrentSchema();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_current_user() returns varchar(200) as
            class GET_CURRENT_USER {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getCurrentUser();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SCALAR SCRIPT
            get_database_name() returns varchar(300) AS
            class GET_DATABASE_NAME {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getDatabaseName();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_database_version() returns varchar(20) as
            class GET_DATABASE_VERSION {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getDatabaseVersion();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_input_column_count_scalar(c1 double, c2 varchar(100))
            returns number as
            class GET_INPUT_COLUMN_COUNT_SCALAR {
                static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getInputColumnCount();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java set script
            get_input_column_count_set(c1 double, c2 varchar(100))
            returns number as
            class GET_INPUT_COLUMN_COUNT_SET {
                static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getInputColumnCount();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
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
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_input_type_scalar() returns varchar(200) as
            class GET_INPUT_TYPE_SCALAR {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getInputType();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java set script
            get_input_type_set(a double) returns varchar(200) as
            class GET_INPUT_TYPE_SET {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getInputType();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_node_id() returns number as
            class GET_NODE_ID {
                static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getNodeId();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_output_column_count_emit()
            emits (x number, y number, z number) as
            class GET_OUTPUT_COLUMN_COUNT_EMIT {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(exa.getOutputColumnCount(), exa.getOutputColumnCount(), exa.getOutputColumnCount());
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_output_column_count_return()
            returns number as
            class GET_OUTPUT_COLUMN_COUNT_RETURN {
                static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getOutputColumnCount();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
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
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_output_type_emit()
            emits (t varchar(200)) as
            class GET_OUTPUT_TYPE_EMIT {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(exa.getOutputType());
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_output_type_return()
            returns varchar(200) as
            class GET_OUTPUT_TYPE_RETURN {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getOutputType();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
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
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_scope_user() returns varchar(200) as
            class GET_SCOPE_USER {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getScopeUser();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_script_code() returns varchar(2000) as
            class GET_SCRIPT_CODE {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getScriptCode();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_script_language() emits (s1 varchar(300), s2 varchar(300)) as
            class GET_SCRIPT_LANGUAGE {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(exa.getScriptLanguage(), "Java");
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_script_name() returns varchar(200) as
            class GET_SCRIPT_NAME {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getScriptName();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_script_schema() returns varchar(200) as
            class GET_SCRIPT_SCHEMA {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getScriptSchema();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_session_id() returns varchar(200) as
            class GET_SESSION_ID {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getSessionId();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_statement_id() returns number as
            class GET_STATEMENT_ID {
                static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getStatementId();
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script
            get_vm_id() returns varchar(200) as
            class GET_VM_ID {
                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return exa.getVmId();
                }
            }
            /
        '''))

class MetaDataTest(_JavaUdfSetup):

    def test_database_name(self):
        rows = self.query('''SELECT fn1.get_database_name() FROM DUAL''')
        self.assertTrue(len(rows[0][0]) > 0)

    def test_database_version(self):
        rows = self.query('''select fn1.get_database_version() from dual''')
        self.assertTrue(len(rows[0][0]) > 0)

    def test_script_language(self):
        rows = self.query('''select fn1.get_script_language() from dual''')
        self.assertTrue((rows[0][0]).upper().startswith((rows[0][1]).upper()))

    def test_script_name(self):
        rows = self.query('''select fn1.get_script_name() from dual''')
        self.assertRowEqual(('GET_SCRIPT_NAME',), rows[0])

    def test_script_schema(self):
        if (udf.opts.is_compat_mode != "true"):
            rows = self.query('''select fn1.get_script_schema() from dual''')
            self.assertRowEqual(('FN1',), rows[0])

    def test_script_user(self):
        if (udf.opts.is_compat_mode != "true"):
            rows = self.query('''select fn1.get_current_user() from dual''')
            self.assertRowEqual(('SYS',), rows[0])

    def test_scope_user(self):
        if (udf.opts.is_compat_mode != "true"):
            rows = self.query('''select fn1.get_scope_user() from dual''')
            self.assertRowEqual(('SYS',), rows[0])

    def test_current_schema_null(self):
        if (udf.opts.is_compat_mode != "true"):
            self.query('''CLOSE SCHEMA''')
            rows = self.query('''select fn1.get_current_schema() from dual''')
            self.assertRowEqual(('NULL',), rows[0])

    def test_current_schema(self):
        if (udf.opts.is_compat_mode != "true"):
            self.query('''create schema test_schema''')
            rows = self.query('''select fn1.get_current_schema() from dual''')
            self.assertRowEqual(('TEST_SCHEMA',), rows[0])
            self.query('''drop schema test_schema cascade''')

    def test_script_code(self):
        rows = self.query('''select fn1.get_script_code() from dual''')
        self.assertTrue((rows[0][0]).upper().find('CTX') >= 0)

    def test_session_id(self):
        rows = self.query('''select fn1.get_session_id() from dual''')
        self.assertTrue(len(rows[0][0]) > 0)

    def test_statement_id(self):
        rows = self.query('''select fn1.get_statement_id() from dual''')
        self.assertTrue(rows[0][0] >= 0)

    def test_node_id(self):
        rows = self.query('''select fn1.get_node_id() from dual''')
        self.assertTrue(rows[0][0] >= 0)

    def test_vm_id(self):
        rows = self.query('''select fn1.get_vm_id() from dual''')
        self.assertTrue(len(rows[0][0]) > 0)

    def test_input_type_scalar(self):
        rows = self.query('''select fn1.get_input_type_scalar() from dual''')
        self.assertRowEqual(('SCALAR',), rows[0])

    def test_input_type_set(self):
        rows = self.query('''select fn1.get_input_type_set(x) from (values 1,2,3) as t(x)''')
        self.assertRowEqual(('SET',), rows[0])

    def test_input_column_count_scalar(self):
        rows = self.query('''select fn1.get_input_column_count_scalar(12.3, 'hihihi') from dual''')
        self.assertRowEqual((2,), rows[0])

    
    def test_input_column_count_set(self):
        rows = self.query('''select fn1.get_input_column_count_set(x, y) from (values (12.3, 'hihihi')) as t(x,y)''')
        self.assertRowEqual((2,), rows[0])

    def test_input_columns(self):
        rows = self.query('''select fn1.get_input_columns(1.2, '123') from dual order by column_id''')
        r0 = rows[0]
        r1 = rows[1]
        self.assertTrue(r0[0] == 1)
        self.assertTrue(r0[1].upper() == 'C1')
        self.assertTrue(r0[2].upper() == 'NUMBER' or r0[2].upper() == "<TYPE 'FLOAT'>" or r0[2].upper() == "DOUBLE" or r0[2].upper() == "JAVA.LANG.DOUBLE" or r0[2].upper() == "<CLASS 'FLOAT'>")
        self.assertTrue(r0[3].upper() == 'DOUBLE')
        self.assertTrue(r1[0] == 2)
        self.assertTrue(r1[1].upper() == 'C2')
        self.assertTrue(r1[2].upper() == 'STRING' or r1[2].upper() == "<TYPE 'UNICODE'>" or r1[2].upper() == "CHARACTER" or r1[2].upper() == "JAVA.LANG.STRING" or r1[2].upper() == "<CLASS 'STR'>")
        self.assertTrue(r1[3].upper().startswith('VARCHAR(200)'))
        self.assertTrue(r0[6] == 0)
        self.assertTrue(r1[6] == 200) 
   
    def test_output_type_return(self):
        rows = self.query('''select fn1.get_output_type_return() from dual''')
        self.assertTrue(rows[0][0] == 'RETURN')

     
    def test_output_type_emit(self):
        rows = self.query('''select fn1.get_output_type_emit() from dual''')
        self.assertTrue(rows[0][0] == 'EMIT')


    def test_output_column_count_return(self):
        rows = self.query('''select fn1.get_output_column_count_return() from dual''')
        self.assertRowEqual((1,),rows[0])


    def test_output_column_count_emit(self):
        rows = self.query('''select fn1.get_output_column_count_emit() from dual''')
        self.assertRowEqual((3,3,3),rows[0])

    def test_output_columns(self):
        rows = self.query('''select fn1.get_output_columns() from dual order by column_id''')
        r0 = rows[0]
        r1 = rows[1]
        r2 = rows[2]
        r3 = rows[3]
        r4 = rows[4]
        r5 = rows[5]
        r6 = rows[6]
        self.assertTrue(r0[0] == 1)
        self.assertTrue(r1[0] == 2)
        self.assertTrue(r2[0] == 3)
        self.assertTrue(r3[0] == 4)
        self.assertTrue(r4[0] == 5)
        self.assertTrue(r5[0] == 6)
        self.assertTrue(r6[0] == 7)
        self.assertTrue(r0[1].upper() == 'COLUMN_ID')
        self.assertTrue(r1[1].upper() == 'COLUMN_NAME')
        self.assertTrue(r2[1].upper() == 'COLUMN_TYPE')
        self.assertTrue(r3[1].upper() == 'COLUMN_SQL_TYPE')
        self.assertTrue(r4[1].upper() == 'COLUMN_PRECISION')
        self.assertTrue(r5[1].upper() == 'COLUMN_SCALE')
        self.assertTrue(r6[1].upper() == 'COLUMN_LENGTH')
        self.assertTrue(r0[2].upper() == 'NUMBER' or r0[2].upper() == "<TYPE 'FLOAT'>" or r0[2].upper() == "DOUBLE" or r0[2].upper() == "JAVA.LANG.DOUBLE" or r0[2].upper() == "<CLASS 'FLOAT'>")
        self.assertTrue(r1[2].upper() == 'STRING' or r1[2].upper() == "<TYPE 'UNICODE'>" or r1[2].upper() == "CHARACTER" or r1[2].upper() == "JAVA.LANG.STRING" or r1[2].upper() == "<CLASS 'STR'>")
        self.assertTrue(r2[2].upper() == 'STRING' or r2[2].upper() == "<TYPE 'UNICODE'>" or r2[2].upper() == "CHARACTER" or r2[2].upper() == "JAVA.LANG.STRING" or r2[2].upper() == "<CLASS 'STR'>")
        self.assertTrue(r3[2].upper() == 'STRING' or r3[2].upper() == "<TYPE 'UNICODE'>" or r3[2].upper() == "CHARACTER" or r3[2].upper() == "JAVA.LANG.STRING" or r3[2].upper() == "<CLASS 'STR'>")
        self.assertTrue(r4[2].upper() == 'NUMBER' or r4[2].upper() == "<TYPE 'FLOAT'>" or r4[2].upper() == "DOUBLE" or r4[2].upper() == "JAVA.LANG.DOUBLE" or r4[2].upper() == "<CLASS 'FLOAT'>")
        self.assertTrue(r5[2].upper() == 'NUMBER' or r5[2].upper() == "<TYPE 'FLOAT'>" or r5[2].upper() == "DOUBLE" or r5[2].upper() == "JAVA.LANG.DOUBLE" or r5[2].upper() == "<CLASS 'FLOAT'>")
        self.assertTrue(r6[2].upper() == 'NUMBER' or r6[2].upper() == "<TYPE 'FLOAT'>" or r6[2].upper() == "DOUBLE" or r6[2].upper() == "JAVA.LANG.DOUBLE" or r6[2].upper() == "<CLASS 'FLOAT'>")
        self.assertTrue(r0[3].upper() == 'DOUBLE')
        self.assertTrue(r1[3].upper().startswith('VARCHAR(200)'))
        self.assertTrue(r2[3].upper().startswith('VARCHAR(20)'))
        self.assertTrue(r3[3].upper().startswith('VARCHAR(20)'))
        self.assertTrue(r4[3].upper() == 'DOUBLE')
        self.assertTrue(r5[3].upper() == 'DOUBLE')
        self.assertTrue(r6[3].upper() == 'DOUBLE')
        self.assertTrue(r1[6] == 200) 
        self.assertTrue(r2[6] == 20) 
        self.assertTrue(r3[6] == 20) 



    def test_precision_scale_length(self):
        rows = self.query('''select fn1.get_precision_scale_length(2.5, '0123456789') from dual''')
        self.assertRowEqual((6,3,0,0,0,10), rows[0])


    def test_char_length(self):
        rows = self.query('''select fn1.get_char_length('0123456789') from dual''')
        self.assertRowEqual((10,20,'9876543210          '), (int(rows[0][0]), int(rows[0][1]), rows[0][2]))


if __name__ == '__main__':
    udf.main()
