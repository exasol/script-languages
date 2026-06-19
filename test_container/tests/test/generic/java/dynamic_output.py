#!/usr/bin/env python3

from exasol_python_test_framework import udf


class _JavaUdfSetup(udf.TestCase):

    def _setup_fn1_schema(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')

    def _setup_data(self):
        self.query('DROP SCHEMA dynamic_output CASCADE', ignore_errors=True)
        self.query('''drop connection SPOT4245''', ignore_errors=True)
        self.query('CREATE SCHEMA dynamic_output')
        self.query('CREATE TABLE dynamic_output.small(x VARCHAR(2000), y DOUBLE)')
        self.query('''INSERT INTO dynamic_output.small VALUES ('Some string ... and some more', 2.2)''')
        self.query('create table dynamic_output.groupt(id int, n double, v varchar(999))')
        self.query('''insert into dynamic_output.groupt values (1,1,'aa'),
                                                (1,2,'ab'),
                                                (2,2,'ba')
                                                ''')
        self.query('create table dynamic_output.target (a int, b double, c varchar(100));')
        self.query('''create connection SPOT4245 to 'a' user 'b' identified by 'c' ''')

    def _create_all_udfs(self):
        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT VAREMIT_SIMPLE_SET (a double) EMITS (...) AS
            class VAREMIT_SIMPLE_SET {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(1.0);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_SIMPLE_SET (a double) EMITS (...) AS
            class DEFAULT_VAREMIT_SIMPLE_SET {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(1.0);
                }
                static String getDefaultOutputColumns() {
                    return "x double";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SCALAR SCRIPT VAREMIT_SIMPLE_SCALAR (a double) EMITS (...) AS
            class VAREMIT_SIMPLE_SCALAR {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(1.0);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SCALAR SCRIPT DEFAULT_VAREMIT_SIMPLE_SCALAR (a double) EMITS (...) AS
            class DEFAULT_VAREMIT_SIMPLE_SCALAR {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(1.0);
                }
                static String getDefaultOutputColumns() {
                    return "x double";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SCALAR SCRIPT VAREMIT_SIMPLE_ALL_DYN (...) EMITS (...) AS
            class VAREMIT_SIMPLE_ALL_DYN {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(1.0);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SCALAR SCRIPT DEFAULT_VAREMIT_SIMPLE_ALL_DYN (...) EMITS (...) AS
            class DEFAULT_VAREMIT_SIMPLE_ALL_DYN {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(1.0);
                }
                static String getDefaultOutputColumns() {
                    return "x double";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT VAREMIT_SIMPLE_SYNTAX_VAR (...) EMITS ( ...   ) AS
            class VAREMIT_SIMPLE_SYNTAX_VAR {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(1.0);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_SIMPLE_SYNTAX_VAR (...) EMITS ( ...   ) AS
            class DEFAULT_VAREMIT_SIMPLE_SYNTAX_VAR {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(1.0);
                }
                static String getDefaultOutputColumns() {
                    return "x double";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT VAREMIT_GENERIC_EMIT (a varchar(100)) EMITS (...) AS
            class VAREMIT_GENERIC_EMIT {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    String val = ctx.getString("a");
                    int n = (int) exa.getOutputColumnCount();
                    Object[] record = new Object[n];
                    for (int i = 0; i < n; i++) {
                        record[i] = val;
                    }
                    ctx.emit(record);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_GENERIC_EMIT (a varchar(100)) EMITS (...) AS
            class DEFAULT_VAREMIT_GENERIC_EMIT {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    String val = ctx.getString("a");
                    int n = (int) exa.getOutputColumnCount();
                    Object[] record = new Object[n];
                    for (int i = 0; i < n; i++) {
                        record[i] = val;
                    }
                    ctx.emit(record);
                }
                static String getDefaultOutputColumns() {
                    return "a varchar(100)";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT VAREMIT_ALL_GENERIC (...) EMITS (...) AS
            class VAREMIT_ALL_GENERIC {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    String val = ctx.getString(0);
                    int n = (int) exa.getOutputColumnCount();
                    Object[] record = new Object[n];
                    for (int i = 0; i < n; i++) {
                        record[i] = val;
                    }
                    ctx.emit(record);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_ALL_GENERIC (...) EMITS (...) AS
            class DEFAULT_VAREMIT_ALL_GENERIC {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    String val = ctx.getString(0);
                    int n = (int) exa.getOutputColumnCount();
                    Object[] record = new Object[n];
                    for (int i = 0; i < n; i++) {
                        record[i] = val;
                    }
                    ctx.emit(record);
                }
                static String getDefaultOutputColumns() {
                    return "a varchar(100)";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT VAREMIT_METADATA_SET_EMIT (...) EMITS(...) AS
            class VAREMIT_METADATA_SET_EMIT {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    long count = exa.getOutputColumnCount();
                    ctx.emit(String.valueOf(count), 1.0);
                    for (int i = 0; i < count; i++) {
                        ctx.emit(exa.getOutputColumnName(i), 1.0);
                        ctx.emit(exa.getOutputColumnType(i).getName(), 1.0);
                        ctx.emit(exa.getOutputColumnSqlType(i), 1.0);
                        ctx.emit(String.valueOf(exa.getOutputColumnPrecision(i)), 1.0);
                        ctx.emit(String.valueOf(exa.getOutputColumnScale(i)), 1.0);
                        ctx.emit(String.valueOf(exa.getOutputColumnLength(i)), 1.0);
                    }
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_METADATA_SET_EMIT (...) EMITS(...) AS
            class DEFAULT_VAREMIT_METADATA_SET_EMIT {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    long count = exa.getOutputColumnCount();
                    ctx.emit(String.valueOf(count), 1.0);
                    for (int i = 0; i < count; i++) {
                        ctx.emit(exa.getOutputColumnName(i), 1.0);
                        ctx.emit(exa.getOutputColumnType(i).getName(), 1.0);
                        ctx.emit(exa.getOutputColumnSqlType(i), 1.0);
                        ctx.emit(String.valueOf(exa.getOutputColumnPrecision(i)), 1.0);
                        ctx.emit(String.valueOf(exa.getOutputColumnScale(i)), 1.0);
                        ctx.emit(String.valueOf(exa.getOutputColumnLength(i)), 1.0);
                    }
                }
                static String getDefaultOutputColumns() {
                    return "a varchar(123), b double";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT VAREMIT_NON_VAR_EMIT (...) EMITS (a double) AS
            class VAREMIT_NON_VAR_EMIT {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(1.0);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_NON_VAR_EMIT (...) EMITS (a double) AS
            class DEFAULT_VAREMIT_NON_VAR_EMIT {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(1.0);
                }
                static String getDefaultOutputColumns() {
                    return "a int";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT VAREMIT_SIMPLE_RETURNS (a int) RETURNS int AS
            class VAREMIT_SIMPLE_RETURNS {
                static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return 1L;
                }
                static String getDefaultOutputColumns() {
                    return "x double";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_SIMPLE_RETURNS (a int) RETURNS int AS
            class DEFAULT_VAREMIT_SIMPLE_RETURNS {
                static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return 1L;
                }
                static String getDefaultOutputColumns() {
                    return "a int";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT VAREMIT_EMIT_INPUT (...) EMITS (...) AS
            import java.math.BigDecimal;
            import java.sql.Date;
            import java.sql.Timestamp;
            class VAREMIT_EMIT_INPUT {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    int n = (int) exa.getInputColumnCount();
                    Object[] record = new Object[n];
                    for (int i = 0; i < n; i++) {
                        Class cls = exa.getInputColumnType(i);
                        if (cls == Integer.class) record[i] = ctx.getInteger(i);
                        else if (cls == Long.class) record[i] = ctx.getLong(i);
                        else if (cls == Double.class) record[i] = ctx.getDouble(i);
                        else if (cls == String.class) record[i] = ctx.getString(i);
                        else if (cls == Boolean.class) record[i] = ctx.getBoolean(i);
                        else if (cls == BigDecimal.class) record[i] = ctx.getBigDecimal(i);
                        else if (cls == Date.class) record[i] = ctx.getDate(i);
                        else if (cls == Timestamp.class) record[i] = ctx.getTimestamp(i);
                    }
                    ctx.emit(record);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_EMIT_INPUT (...) EMITS (...) AS
            import java.math.BigDecimal;
            import java.sql.Date;
            import java.sql.Timestamp;
            class DEFAULT_VAREMIT_EMIT_INPUT {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    int n = (int) exa.getInputColumnCount();
                    Object[] record = new Object[n];
                    for (int i = 0; i < n; i++) {
                        Class cls = exa.getInputColumnType(i);
                        if (cls == Integer.class) record[i] = ctx.getInteger(i);
                        else if (cls == Long.class) record[i] = ctx.getLong(i);
                        else if (cls == Double.class) record[i] = ctx.getDouble(i);
                        else if (cls == String.class) record[i] = ctx.getString(i);
                        else if (cls == Boolean.class) record[i] = ctx.getBoolean(i);
                        else if (cls == BigDecimal.class) record[i] = ctx.getBigDecimal(i);
                        else if (cls == Date.class) record[i] = ctx.getDate(i);
                        else if (cls == Timestamp.class) record[i] = ctx.getTimestamp(i);
                    }
                    ctx.emit(record);
                }
                static String getDefaultOutputColumns() {
                    return "a int";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT VAREMIT_EMIT_INPUT_WITH_META_CHECK (...) EMITS (...) AS
            import java.math.BigDecimal;
            import java.sql.Date;
            import java.sql.Timestamp;
            class VAREMIT_EMIT_INPUT_WITH_META_CHECK {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    int n = (int) exa.getInputColumnCount();
                    Object[] record = new Object[n];
                    for (int i = 0; i < n; i++) {
                        if (!exa.getInputColumnSqlType(i).equals(exa.getOutputColumnSqlType(i))) {
                            throw new Exception("Input/output SQL type mismatch at column " + i);
                        }
                        Class cls = exa.getInputColumnType(i);
                        if (cls == Integer.class) record[i] = ctx.getInteger(i);
                        else if (cls == Long.class) record[i] = ctx.getLong(i);
                        else if (cls == Double.class) record[i] = ctx.getDouble(i);
                        else if (cls == String.class) record[i] = ctx.getString(i);
                        else if (cls == Boolean.class) record[i] = ctx.getBoolean(i);
                        else if (cls == BigDecimal.class) record[i] = ctx.getBigDecimal(i);
                        else if (cls == Date.class) record[i] = ctx.getDate(i);
                        else if (cls == Timestamp.class) record[i] = ctx.getTimestamp(i);
                    }
                    ctx.emit(record);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_EMIT_INPUT_WITH_META_CHECK (...) EMITS (...) AS
            import java.math.BigDecimal;
            import java.sql.Date;
            import java.sql.Timestamp;
            class DEFAULT_VAREMIT_EMIT_INPUT_WITH_META_CHECK {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    int n = (int) exa.getInputColumnCount();
                    Object[] record = new Object[n];
                    for (int i = 0; i < n; i++) {
                        if (!exa.getInputColumnSqlType(i).equals(exa.getOutputColumnSqlType(i))) {
                            throw new Exception("Input/output SQL type mismatch at column " + i);
                        }
                        Class cls = exa.getInputColumnType(i);
                        if (cls == Integer.class) record[i] = ctx.getInteger(i);
                        else if (cls == Long.class) record[i] = ctx.getLong(i);
                        else if (cls == Double.class) record[i] = ctx.getDouble(i);
                        else if (cls == String.class) record[i] = ctx.getString(i);
                        else if (cls == Boolean.class) record[i] = ctx.getBoolean(i);
                        else if (cls == BigDecimal.class) record[i] = ctx.getBigDecimal(i);
                        else if (cls == Date.class) record[i] = ctx.getDate(i);
                        else if (cls == Timestamp.class) record[i] = ctx.getTimestamp(i);
                    }
                    ctx.emit(record);
                }
                static String getDefaultOutputColumns() {
                    return "a varchar(123), b double";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT DEFAULT_VAREMIT_EMPTY_DEF(X DOUBLE) EMITS (...) AS
            class DEFAULT_VAREMIT_EMPTY_DEF {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(1.4);
                }
                static String getDefaultOutputColumns() {
                    return "";
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT COPY_RELATION (...) EMITS (...) AS
            import java.math.BigDecimal;
            import java.sql.Date;
            import java.sql.Timestamp;
            class COPY_RELATION {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    int n = (int) exa.getInputColumnCount();
                    Object[] record = new Object[n];
                    for (int i = 0; i < n; i++) {
                        if (!exa.getInputColumnSqlType(i).equals(exa.getOutputColumnSqlType(i))) {
                            throw new Exception("Input/output SQL type mismatch at column " + i);
                        }
                        Class cls = exa.getInputColumnType(i);
                        if (cls == Integer.class) record[i] = ctx.getInteger(i);
                        else if (cls == Long.class) record[i] = ctx.getLong(i);
                        else if (cls == Double.class) record[i] = ctx.getDouble(i);
                        else if (cls == String.class) record[i] = ctx.getString(i);
                        else if (cls == Boolean.class) record[i] = ctx.getBoolean(i);
                        else if (cls == BigDecimal.class) record[i] = ctx.getBigDecimal(i);
                        else if (cls == Date.class) record[i] = ctx.getDate(i);
                        else if (cls == Timestamp.class) record[i] = ctx.getTimestamp(i);
                    }
                    ctx.emit(record);
                }
                static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < exa.getInputColumnCount(); i++) {
                        if (i > 0) sb.append(", ");
                        String colName = exa.getInputColumnName(i);
                        try {
                            int idx = Integer.parseInt(colName);
                            colName = "col_" + idx;
                        } catch (NumberFormatException e) {}
                        sb.append(colName).append(" ").append(exa.getInputColumnSqlType(i));
                    }
                    return sb.toString();
                }
            }
            /
        '''))

    def setUp(self):
        self._setup_fn1_schema()
        self._setup_data()
        self.query('OPEN SCHEMA FN1')
        self._create_all_udfs()


class DynamicOutputCreateScript(_JavaUdfSetup):

    def test_create_script_set(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='VAREMIT_SIMPLE_SET' and SCRIPT_TEXT LIKE 'CREATE % SET SCRIPT "VAREMIT_SIMPLE_SET" ("a" DOUBLE) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])

    def test_create_script_scalar(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='VAREMIT_SIMPLE_SCALAR' and SCRIPT_TEXT LIKE 'CREATE % SCALAR SCRIPT "VAREMIT_SIMPLE_SCALAR" ("a" DOUBLE) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])

    def test_create_script_all_dyn(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='VAREMIT_SIMPLE_ALL_DYN' and SCRIPT_TEXT LIKE 'CREATE % SCALAR SCRIPT "VAREMIT_SIMPLE_ALL_DYN" (...) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])

    def test_create_script_syntax_var(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='VAREMIT_SIMPLE_SYNTAX_VAR' and SCRIPT_TEXT LIKE 'CREATE % SET SCRIPT "VAREMIT_SIMPLE_SYNTAX_VAR" (...) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])


class DynamicOutputTest(_JavaUdfSetup):

    def test_generic_emit(self):
        rows = self.query('''
            SELECT fn1.VAREMIT_GENERIC_EMIT('SUPERDYNAMIC') EMITS (a varchar(100));
            ''')
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])

    def test_all_generic(self):
        rows = self.query('''
            SELECT fn1.VAREMIT_ALL_GENERIC('SUPERDYNAMIC') EMITS (a varchar(100));
            ''')
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])

    def test_correctness_emits_subquery(self):
        rows = self.query('''
            SELECT "A" || 'x' || "B" FROM (
            SELECT fn1.VAREMIT_GENERIC_EMIT('SUPERDYNAMIC') EMITS (a varchar(100), b varchar(100)));
            ''')
        self.assertRowEqual(('SUPERDYNAMICxSUPERDYNAMIC',), rows[0])

    def test_correctness_emits_with_grouping(self):
        rows = self.query('''
            SELECT 'X' || count(a) || 'X' FROM (
             SELECT fn1.VAREMIT_GENERIC_EMIT('SUPERDYNAMIC') EMITS (a varchar(100))
             FROM dynamic_output.groupt GROUP BY id
            ) where a = 'SUPERDYNAMIC';
            ''')
        self.assertRowEqual(('X2X',), rows[0])

    def test_correctness_nested(self):
        rows = self.query('''
            SELECT fn1.VAREMIT_GENERIC_EMIT(c || 'D') EMITS (d varchar(100)) FROM (
              SELECT fn1.VAREMIT_GENERIC_EMIT(b || 'C') EMITS (c varchar(100)) FROM (
                SELECT fn1.VAREMIT_GENERIC_EMIT(a || 'B') EMITS(b varchar(100)) FROM (
                  SELECT fn1.VAREMIT_GENERIC_EMIT('A') EMITS (a varchar(100))
                )
              )
            );
            ''')
        self.assertRowEqual(('ABCD',), rows[0])

    def test_metadata_correctness(self):
        rows = self.query('''
            SELECT fn1.VAREMIT_METADATA_SET_EMIT(1) EMITS (a varchar(123), b double)
            FROM DUAL
            ''')
        self.assertRowEqual(('2', 1.0), rows[0])
        self.assertRowEqual(('A', 1.0), rows[1])
        self.assertTrue(rows[2][0] in ["java.lang.String"])
        self.assertRowEqual(('VARCHAR(123) UTF8', 1), rows[3])
        self.assertRowEqual(('123', 1.0), rows[6])
        self.assertRowEqual(('B', 1.0), rows[7])
        self.assertTrue(rows[8][0] in ["java.lang.Double"])
        self.assertRowEqual(('DOUBLE', 1.0), rows[9])


class DynamicOutputWrongUsage(_JavaUdfSetup):

    def test_error_emit_missing(self):
        with self.assertRaisesRegex(Exception, 'The script has dynamic return arguments. Either specify the return arguments in the query via EMITS or implement the method'):
            self.query('''SELECT fn1.VAREMIT_GENERIC_EMIT(1)''')

    def test_error_empty_emit(self):
        with self.assertRaisesRegex(Exception, 'Empty return argument definition is not allowed'):
            self.query('''SELECT fn1.VAREMIT_GENERIC_EMIT(1) EMITS ();''')

    def test_error_empty_emit_2(self):
        with self.assertRaisesRegex(Exception, 'syntax error'):
            self.query('''SELECT fn1.VAREMIT_GENERIC_EMIT(1) EMITS (a);''')

    def test_error_wrong_emit(self):
        with self.assertRaisesRegex(Exception, 'syntax error'):
            self.query('''SELECT fn1.VAREMIT_GENERIC_EMIT(1) EMITS (int);''')

    def test_error_redundant_name(self):
        with self.assertRaisesRegex(Exception, 'Return argument A is declared more than once'):
            self.query('''SELECT fn1.VAREMIT_GENERIC_EMIT(1) EMITS (a int, b int, a int);''')

    def test_error_non_var_emit(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition. Dynamic return arguments are not supported in this case'):
            self.query('''SELECT fn1.VAREMIT_NON_VAR_EMIT(1) EMITS (a double);''')

    def test_error_non_var_emit_2(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition. Dynamic return arguments are not supported in this case'):
            self.query('''SELECT fn1.VAREMIT_NON_VAR_EMIT(1) EMITS ();''')

    def test_error_returns_not_supported(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition. Dynamic return arguments are not supported in this case'):
            self.query('''select fn1.VAREMIT_SIMPLE_RETURNS(1) EMITS (a INT);''')

    def test_error_built_in_set_not_supported(self):
        with self.assertRaisesRegex(Exception, 'emits specification is not allowed for built-in functions'):
            self.query('''SELECT AVG(a) EMITS(a int) FROM dynamic_output.small;''')

    def test_error_built_in_scalar_not_supported(self):
        with self.assertRaisesRegex(Exception, 'emits specification is not allowed for built-in functions'):
            self.query('''SELECT -ABS(a) EMITS(a int) FROM dynamic_output.small;''')


class DynamicOutputInsertInto(_JavaUdfSetup):

    def test_insert_basic(self):
        self.query('''delete from dynamic_output.target;''')
        self.query('''
            insert into dynamic_output.target select fn1.VAREMIT_EMIT_INPUT(1, CAST (1.1 AS DOUBLE), 'a');
            ''')
        rows = self.query('''
            select * from dynamic_output.target;
            ''')
        self.assertRowEqual((1, 1.1, 'a'), rows[0])
        self.query('''delete from dynamic_output.target;''')

    def test_insert_metadata_correctness(self):
        self.query('''
            insert into dynamic_output.target select fn1.VAREMIT_EMIT_INPUT_WITH_META_CHECK(cast (2 as int), CAST (2.2 AS DOUBLE), cast ('b' as varchar(100)));
            ''')
        rows = self.query('''
            select * from dynamic_output.target;
            ''')
        self.assertRowEqual((2, 2.2, 'b'), rows[0])
        self.query('''delete from dynamic_output.target;''')

    def test_insert_target_columns_change_order(self):
        self.query('''
            insert into dynamic_output.target (c, b, a) select fn1.VAREMIT_EMIT_INPUT('c', CAST (3.3 AS DOUBLE), 3);
            ''')
        rows = self.query('''
            select * from dynamic_output.target;
            ''')
        self.assertRowEqual((3, 3.3, 'c'), rows[0])
        self.query('''delete from dynamic_output.target;''')

    def test_insert_target_columns_subset(self):
        self.query('''
            insert into dynamic_output.target (b) select fn1.VAREMIT_EMIT_INPUT(CAST (4.4 AS DOUBLE));
            ''')
        rows = self.query('''
            select * from dynamic_output.target;
            ''')
        self.assertRowEqual((None, 4.4, None), rows[0])
        self.query('''delete from dynamic_output.target;''')

    def test_insert_emits_not_allowed(self):
        with self.assertRaisesRegex(Exception, 'The return arguments for EMITS functions are inferred from the table to insert into. Specification of EMITS is not allowed in this case.'):
            self.query('''insert into dynamic_output.target select FN1.VAREMIT_EMIT_INPUT(1) emits (a int);''')


class DynamicOutputCreateTableAs(_JavaUdfSetup):

    def test_insert_basic(self):
        self.query('''drop table if exists dynamic_output.targetcreated;''')
        self.query('''
            create table dynamic_output.targetcreated as select fn1.VAREMIT_EMIT_INPUT(1, CAST (1.1 AS DOUBLE), 'a') emits (a decimal(20,0), b double, c varchar(100));
            ''')
        rows = self.query('''
            select * from dynamic_output.targetcreated;
            ''')
        self.assertRowEqual((1, 1.1, 'a'), rows[0])
        rows = self.query('''
            describe dynamic_output.targetcreated;
            ''')
        self.assertRowEqual(('A', 'DECIMAL(20,0)'), rows[0][0:2])
        self.assertRowEqual(('B', 'DOUBLE'), rows[1][0:2])
        self.assertRowEqual(('C', 'VARCHAR(100) UTF8'), rows[2][0:2])


## #####################################################
## The same as above but now with default output columns
## #####################################################

class DynamicOutputUDFsExistInExaAllScripts(_JavaUdfSetup):

    def test_create_script_set(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='DEFAULT_VAREMIT_SIMPLE_SET' and SCRIPT_TEXT LIKE 'CREATE % SET SCRIPT "DEFAULT_VAREMIT_SIMPLE_SET" ("a" DOUBLE) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])

    def test_create_script_scalar(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='DEFAULT_VAREMIT_SIMPLE_SCALAR' and SCRIPT_TEXT LIKE 'CREATE % SCALAR SCRIPT "DEFAULT_VAREMIT_SIMPLE_SCALAR" ("a" DOUBLE) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])

    def test_create_script_all_dyn(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='DEFAULT_VAREMIT_SIMPLE_ALL_DYN' and SCRIPT_TEXT LIKE 'CREATE % SCALAR SCRIPT "DEFAULT_VAREMIT_SIMPLE_ALL_DYN" (...) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])

    def test_create_script_syntax_var(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='DEFAULT_VAREMIT_SIMPLE_SYNTAX_VAR' and SCRIPT_TEXT LIKE 'CREATE % SET SCRIPT "DEFAULT_VAREMIT_SIMPLE_SYNTAX_VAR" (...) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])


class DefaultDynamicOutputTest(_JavaUdfSetup):

    def test_generic_emit(self):
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT('SUPERDYNAMIC') EMITS (a varchar(100));
            ''')
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT('SUPERDYNAMIC');
            ''')
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])

    def test_all_generic(self):
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_ALL_GENERIC('SUPERDYNAMIC') EMITS (a varchar(100));
            ''')
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_ALL_GENERIC('SUPERDYNAMIC');
            ''')
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])

    def test_correctness_emits_subquery(self):
        rows = self.query('''
            SELECT "A" || 'x' || "B" FROM (
            SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT('SUPERDYNAMIC') EMITS (a varchar(100), b varchar(100)));
            ''')
        self.assertRowEqual(('SUPERDYNAMICxSUPERDYNAMIC',), rows[0])
        rows = self.query('''
            SELECT "A" || 'x' || "A" FROM (
            SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT('SUPERDYNAMIC'));
            ''')
        self.assertRowEqual(('SUPERDYNAMICxSUPERDYNAMIC',), rows[0])

    def test_correctness_emits_with_grouping(self):
        rows = self.query('''
            SELECT 'X' || count(a) || 'X' FROM (
             SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT('SUPERDYNAMIC') EMITS (a varchar(100))
             FROM dynamic_output.groupt GROUP BY id
            ) where a = 'SUPERDYNAMIC';
            ''')
        self.assertRowEqual(('X2X',), rows[0])
        rows = self.query('''
            SELECT 'X' || count(a) || 'X' FROM (
             SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT('SUPERDYNAMIC')
             FROM dynamic_output.groupt GROUP BY id
            ) where a = 'SUPERDYNAMIC';
            ''')
        self.assertRowEqual(('X2X',), rows[0])

    def test_correctness_nested(self):
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(c || 'D') EMITS (d varchar(100)) FROM (
              SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(b || 'C') EMITS (c varchar(100)) FROM (
                SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(a || 'B') EMITS(b varchar(100)) FROM (
                  SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT('A') EMITS (a varchar(100))
                )
              )
            );
            ''')
        self.assertRowEqual(('ABCD',), rows[0])
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(a || 'D') FROM (
              SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(a || 'C') FROM (
                SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(a || 'B') FROM (
                  SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT('A')
                )
              )
            );
            ''')
        self.assertRowEqual(('ABCD',), rows[0])

    def test_metadata_correctness(self):
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_METADATA_SET_EMIT(1) EMITS (a varchar(123), b double)
            FROM DUAL
            ''')
        self.assertRowEqual(('2', 1.0), rows[0])
        self.assertRowEqual(('A', 1.0), rows[1])
        self.assertTrue(rows[2][0] in ["java.lang.String"])
        self.assertRowEqual(('VARCHAR(123) UTF8', 1), rows[3])
        self.assertRowEqual(('123', 1.0), rows[6])
        self.assertRowEqual(('B', 1.0), rows[7])
        self.assertTrue(rows[8][0] in ["java.lang.Double"])
        self.assertRowEqual(('DOUBLE', 1.0), rows[9])
        # now again with intrinsic emits clause
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_METADATA_SET_EMIT(1)
            FROM DUAL
            ''')
        self.assertRowEqual(('2', 1.0), rows[0])
        self.assertRowEqual(('A', 1.0), rows[1])
        self.assertTrue(rows[2][0] in ["java.lang.String"])
        self.assertRowEqual(('VARCHAR(123) UTF8', 1), rows[3])
        self.assertRowEqual(('123', 1.0), rows[6])
        self.assertRowEqual(('B', 1.0), rows[7])
        self.assertTrue(rows[8][0] in ["java.lang.Double"])
        self.assertRowEqual(('DOUBLE', 1.0), rows[9])


class DefaultDynamicOutputWrongUsage(_JavaUdfSetup):

    def test_error_empty_emit(self):
        with self.assertRaisesRegex(Exception, 'Empty return argument definition is not allowed'):
            self.query('''SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(1) EMITS ();''')

    def test_error_empty_emit_2(self):
        with self.assertRaisesRegex(Exception, 'syntax error'):
            self.query('''SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(1) EMITS (a);''')

    def test_error_wrong_emit(self):
        with self.assertRaisesRegex(Exception, 'syntax error'):
            self.query('''SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(1) EMITS (int);''')

    def test_error_redundant_name(self):
        with self.assertRaisesRegex(Exception, 'Return argument A is declared more than once'):
            self.query('''SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(1) EMITS (a int, b int, a int);''')

    def test_error_non_var_emit(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition. Dynamic return arguments are not supported in this case'):
            self.query('''SELECT fn1.DEFAULT_VAREMIT_NON_VAR_EMIT(1) EMITS (a double);''')

    def test_error_non_var_emit_2(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition. Dynamic return arguments are not supported in this case'):
            self.query('''SELECT fn1.DEFAULT_VAREMIT_NON_VAR_EMIT(1) EMITS ();''')

    def test_error_returns_not_supported(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition. Dynamic return arguments are not supported in this case'):
            self.query('''select fn1.DEFAULT_VAREMIT_SIMPLE_RETURNS(1) EMITS (a INT);''')


class DefaultDynamicOutputInsertInto(_JavaUdfSetup):

    def test_insert_basic(self):
        self.query('''delete from dynamic_output.target;''')
        self.query('''
            insert into dynamic_output.target select fn1.DEFAULT_VAREMIT_EMIT_INPUT(1, CAST (1.1 AS DOUBLE), 'a');
            ''')
        rows = self.query('''
            select * from dynamic_output.target;
            ''')
        self.assertRowEqual((1, 1.1, 'a'), rows[0])
        self.query('''delete from dynamic_output.target;''')

    def test_insert_metadata_correctness(self):
        self.query('''
            insert into dynamic_output.target select fn1.DEFAULT_VAREMIT_EMIT_INPUT_WITH_META_CHECK(cast (2 as int), CAST (2.2 AS DOUBLE), cast ('b' as varchar(100)));
            ''')
        rows = self.query('''
            select * from dynamic_output.target;
            ''')
        self.assertRowEqual((2, 2.2, 'b'), rows[0])
        self.query('''delete from dynamic_output.target;''')

    def test_insert_target_columns_change_order(self):
        self.query('''
            insert into dynamic_output.target (c, b, a) select fn1.DEFAULT_VAREMIT_EMIT_INPUT('c', CAST (3.3 AS DOUBLE), 3);
            ''')
        rows = self.query('''
            select * from dynamic_output.target;
            ''')
        self.assertRowEqual((3, 3.3, 'c'), rows[0])
        self.query('''delete from dynamic_output.target;''')

    def test_insert_target_columns_subset(self):
        self.query('''
            insert into dynamic_output.target (b) select fn1.DEFAULT_VAREMIT_EMIT_INPUT(CAST (4.4 AS DOUBLE));
            ''')
        rows = self.query('''
            select * from dynamic_output.target;
            ''')
        self.assertRowEqual((None, 4.4, None), rows[0])
        self.query('''delete from dynamic_output.target;''')

    def test_insert_emits_not_allowed(self):
        with self.assertRaisesRegex(Exception, 'The return arguments for EMITS functions are inferred from the table to insert into. Specification of EMITS is not allowed in this case.'):
            self.query('''insert into dynamic_output.target select FN1.DEFAULT_VAREMIT_EMIT_INPUT(1) emits (a int);''')


class DefaultDynamicOutputCreateTableAs(_JavaUdfSetup):

    def test_insert_basic(self):
        self.query('''drop table if exists dynamic_output.targetcreated;''')
        self.query('''
            create table dynamic_output.targetcreated as select fn1.DEFAULT_VAREMIT_EMIT_INPUT(1, CAST (1.1 AS DOUBLE), 'a') emits (a decimal(20,0), b double, c varchar(100));
            ''')
        rows = self.query('''
            select * from dynamic_output.targetcreated;
            ''')
        self.assertRowEqual((1, 1.1, 'a'), rows[0])
        rows = self.query('''
            describe dynamic_output.targetcreated;
            ''')
        self.assertRowEqual(('A', 'DECIMAL(20,0)'), rows[0][0:2])
        self.assertRowEqual(('B', 'DOUBLE'), rows[1][0:2])
        self.assertRowEqual(('C', 'VARCHAR(100) UTF8'), rows[2][0:2])


class DefaultDynamicOutputEmptyStringResult(_JavaUdfSetup):

    def test_empty_string_error(self):
        with self.assertRaisesRegex(Exception, 'Empty default output columns'):
            self.query('''select fn1.DEFAULT_VAREMIT_EMPTY_DEF(42.42);''')
        rows = self.query('''select fn1.DEFAULT_VAREMIT_EMPTY_DEF(42.42) emits (x double);''')
        self.assertRowEqual((1.4,), rows[0])


class DefaultDynamicOutputFromInputMeta(_JavaUdfSetup):

    def test_copy_relation(self):
        rows = self.query('''
            select fn1.copy_relation(1,2,3);
            ''')
        self.assertRowEqual((1, 2, 3), rows[0])


if __name__ == '__main__':
    udf.main()
