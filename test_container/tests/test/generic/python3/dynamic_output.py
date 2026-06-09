#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework import exatest


class _Python3UdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
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
        
        self.query('OPEN SCHEMA FN1')
        # Create all Python3 UDFs for dynamic output testing
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT VAREMIT_SIMPLE_SET (a double) EMITS (...) AS
            def run(ctx):
                ctx.emit(1)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT DEFAULT_VAREMIT_SIMPLE_SET (a double) EMITS (...) AS
            def run(ctx):
                ctx.emit(1)
            def default_output_columns():
                return "x double"
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT VAREMIT_SIMPLE_SCALAR (a double) EMITS (...) AS
            def run(ctx):
                ctx.emit(1)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT DEFAULT_VAREMIT_SIMPLE_SCALAR (a double) EMITS (...) AS
            def run(ctx):
                ctx.emit(1)
            def default_output_columns():
                return "x double"
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT VAREMIT_SIMPLE_ALL_DYN (...) EMITS (...) AS
            def run(ctx):
                ctx.emit(1)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT DEFAULT_VAREMIT_SIMPLE_ALL_DYN (...) EMITS (...) AS
            def run(ctx):
                ctx.emit(1)
            def default_output_columns():
                return "x double"
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT VAREMIT_SIMPLE_SYNTAX_VAR (...) EMITS ( ...   ) AS
            def run(ctx):
                ctx.emit(1)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT DEFAULT_VAREMIT_SIMPLE_SYNTAX_VAR (...) EMITS ( ...   ) AS
            def run(ctx):
                ctx.emit(1)
            def default_output_columns():
                return "x double"
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT VAREMIT_GENERIC_EMIT (a varchar(100)) EMITS (...) AS
            def run(ctx):
                ctx.emit(*([ctx[0]]*exa.meta.output_column_count))
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT DEFAULT_VAREMIT_GENERIC_EMIT (a varchar(100)) EMITS (...) AS
            def run(ctx):
                ctx.emit(*([ctx[0]]*exa.meta.output_column_count))
            def default_output_columns():
                return "a varchar(100)"
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT VAREMIT_ALL_GENERIC (...) EMITS (...) AS
            def run(ctx):
                ctx.emit(*([ctx[0]]*exa.meta.output_column_count))
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT DEFAULT_VAREMIT_ALL_GENERIC (...) EMITS (...) AS
            def run(ctx):
                ctx.emit(*([ctx[0]]*exa.meta.output_column_count))
            def default_output_columns():
                return "a varchar(100)"
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT VAREMIT_METADATA_SET_EMIT (...) EMITS(...) AS
            def run(ctx):
                ctx.emit(repr(exa.meta.output_column_count), 1)
                for i in range (0,exa.meta.output_column_count):
                    ctx.emit(exa.meta.output_columns[i].name, 1)
                    ctx.emit(repr(exa.meta.output_columns[i].type), 1)
                    ctx.emit(exa.meta.output_columns[i].sql_type, 1)
                    ctx.emit(repr(exa.meta.output_columns[i].precision), 1)
                    ctx.emit(repr(exa.meta.output_columns[i].scale), 1)
                    ctx.emit(repr(exa.meta.output_columns[i].length), 1)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT DEFAULT_VAREMIT_METADATA_SET_EMIT (...) EMITS(...) AS
            def run(ctx):
                ctx.emit(repr(exa.meta.output_column_count), 1)
                for i in range (0,exa.meta.output_column_count):
                    ctx.emit(exa.meta.output_columns[i].name, 1)
                    ctx.emit(repr(exa.meta.output_columns[i].type), 1)
                    ctx.emit(exa.meta.output_columns[i].sql_type, 1)
                    ctx.emit(repr(exa.meta.output_columns[i].precision), 1)
                    ctx.emit(repr(exa.meta.output_columns[i].scale), 1)
                    ctx.emit(repr(exa.meta.output_columns[i].length), 1)
            def default_output_columns():
                return "a varchar(123), b double"
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT VAREMIT_NON_VAR_EMIT (...) EMITS (a double) AS
            def run(ctx):
                ctx.emit(1)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT DEFAULT_VAREMIT_NON_VAR_EMIT (...) EMITS (a double) AS
            def run(ctx):
                ctx.emit(1)
            def default_output_columns():
                return "a int"
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT VAREMIT_SIMPLE_RETURNS (a int) RETURNS int AS
            def run(ctx):
                return 1
            def default_output_columns():
                return "x double"
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT DEFAULT_VAREMIT_SIMPLE_RETURNS (a int) RETURNS int AS
            def run(ctx):
                return 1
            def default_output_columns():
                return "a int"
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT VAREMIT_EMIT_INPUT (...) EMITS (...) AS
            def run(ctx):
                record = list()
                for col in range(0,exa.meta.input_column_count):
                    record.append(ctx[col])
                ctx.emit(*record)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT DEFAULT_VAREMIT_EMIT_INPUT (...) EMITS (...) AS
            def run(ctx):
                record = list()
                for col in range(0,exa.meta.input_column_count):
                    record.append(ctx[col])
                ctx.emit(*record)
            def default_output_columns():
                return "a int"
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT VAREMIT_EMIT_INPUT_WITH_META_CHECK (...) EMITS (...) AS
            def run(ctx):
                record = list()
                for col in range(0,exa.meta.input_column_count):
                    assert exa.meta.input_columns[col].sql_type == exa.meta.output_columns[col].sql_type
                    record.append(ctx[col])
                ctx.emit(*record)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT DEFAULT_VAREMIT_EMIT_INPUT_WITH_META_CHECK (...) EMITS (...) AS
            def run(ctx):
                record = list()
                for col in range(0,exa.meta.input_column_count):
                    assert exa.meta.input_columns[col].sql_type == exa.meta.output_columns[col].sql_type
                    record.append(ctx[col])
                ctx.emit(*record)
            def default_output_columns():
                return "a varchar(123), b double"
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT DEFAULT_VAREMIT_EMPTY_DEF(X DOUBLE) EMITS (...) AS
            def run(ctx):
                ctx.emit(1.4)

            def default_output_columns():
               return ''
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT COPY_RELATION (...) EMITS (...) AS
            def run(ctx):
                record = list()
                for col in range(0,exa.meta.input_column_count):
                    assert exa.meta.input_columns[col].sql_type == exa.meta.output_columns[col].sql_type
                    record.append(ctx[col])
                ctx.emit(*record)

            def default_output_columns():
                res = list()
                for col in range(0,exa.meta.input_column_count):
                    col_name = exa.meta.input_columns[col].name
                    try:
                        col_name = "col_%s" % (int(col_name))
                    except ValueError: pass
                    res.append("%s %s" % (col_name, exa.meta.input_columns[col].sql_type))
                return ",".join(res)
            /
        '''))


class DynamicOutputCreateScript(_Python3UdfSetup):
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


class DynamicOutputTest(_Python3UdfSetup):
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
        stringType = {
            'python3': ["<type 'unicode'>", "<class 'str'>"],
            'r': ["character"],
            'lua': ["string"],
            'java': ["java.lang.String"]
        }
        numType = {
            'python3': ["<type 'float'>", "<class 'float'>"],
            'r': ["double"],
            'lua': ["number"],
            'java': ["java.lang.Double"]
        }
        self.assertRowEqual(('2',1.0), rows[0])
        self.assertRowEqual(('A',1.0), rows[1])
        self.assertTrue(rows[2][0] in stringType.get(udf.opts.lang or 'python3'))
        self.assertRowEqual(('VARCHAR(123) UTF8',1), rows[3])
        self.assertRowEqual(('123',1.0), rows[6])
        self.assertRowEqual(('B',1.0), rows[7])
        self.assertTrue(rows[8][0] in numType.get(udf.opts.lang or 'python3'))
        self.assertRowEqual(('DOUBLE',1.0), rows[9])


class DynamicOutputWrongUsage(_Python3UdfSetup):
    def test_error_emit_missing(self):
        #with self.assertRaisesRegex(Exception, 'The script has dynamic return arguments. Either specify the return arguments in the query via EMITS or implement the method (default_output_columns|getDefaultOutputColumns|defaultOutputColumns) in the UDF'):
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
            self.query('''SELECT AVG(a) EMITS(a int) FROM VAREMITS;''')

    def test_error_built_in_scalar_not_supported(self):
        with self.assertRaisesRegex(Exception, 'emits specification is not allowed for built-in functions'):
            self.query('''SELECT -ABS(a) EMITS(a int) FROM VAREMITS;''')


class DynamicOutputInsertInto(_Python3UdfSetup):
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


class DynamicOutputCreateTableAs(_Python3UdfSetup):
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

class DefaultDynamicOutputCreateScript(_Python3UdfSetup):
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


class DefaultDynamicOutputTest(_Python3UdfSetup):
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
        stringType = {
            'python3': ["<type 'unicode'>", "<class 'str'>"],
            'r': ["character"],
            'lua': ["string"],
            'java': ["java.lang.String"]
        }
        numType = {
            'python3': ["<type 'float'>", "<class 'float'>"],
            'r': ["double"],
            'lua': ["number"],
            'java': ["java.lang.Double"]
        }
        self.assertRowEqual(('2',1.0), rows[0])
        self.assertRowEqual(('A',1.0), rows[1])
        self.assertTrue(rows[2][0] in stringType.get(udf.opts.lang or 'python3'))
        self.assertRowEqual(('VARCHAR(123) UTF8',1), rows[3])
        self.assertRowEqual(('123',1.0), rows[6])
        self.assertRowEqual(('B',1.0), rows[7])
        self.assertTrue(rows[8][0] in numType.get(udf.opts.lang or 'python3'))
        self.assertRowEqual(('DOUBLE',1.0), rows[9])
        # now again with intrinsic emits clause
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_METADATA_SET_EMIT(1) 
            FROM DUAL
            ''')
        self.assertRowEqual(('2',1.0), rows[0])
        self.assertRowEqual(('A',1.0), rows[1])
        self.assertTrue(rows[2][0] in stringType.get(udf.opts.lang or 'python3'))
        self.assertRowEqual(('VARCHAR(123) UTF8',1), rows[3])
        self.assertRowEqual(('123',1.0), rows[6])
        self.assertRowEqual(('B',1.0), rows[7])
        self.assertTrue(rows[8][0] in numType.get(udf.opts.lang or 'python3'))
        self.assertRowEqual(('DOUBLE',1.0), rows[9])


class DefaultDynamicOutputWrongUsage(_Python3UdfSetup):
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


class DefaultDynamicOutputInsertInto(_Python3UdfSetup):
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


class DefaultDynamicOutputCreateTableAs(_Python3UdfSetup):
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


class DefaultDynamicOutputEmptyStringResult(_Python3UdfSetup):
    def test_empty_string_error(self):
        with self.assertRaisesRegex(Exception, 'Empty default output columns'):
            self.query('''select fn1.DEFAULT_VAREMIT_EMPTY_DEF(42.42);''')
        rows = self.query('''select fn1.DEFAULT_VAREMIT_EMPTY_DEF(42.42) emits (x double);''')
        self.assertRowEqual((1.4,), rows[0])


class DefaultDynamicOutputFromInputMeta(_Python3UdfSetup):
    def test_copy_relation(self):
        rows = self.query('''
            select fn1.copy_relation(1,2,3);
            ''')
        self.assertRowEqual((1, 2, 3), rows[0])


if __name__ == '__main__':
    udf.main()
