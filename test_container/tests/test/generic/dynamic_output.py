#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import requires
from exasol_python_test_framework import exatest


class Test(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA dynamic_output CASCADE', ignore_errors=True)
        self.query('''drop connection SPOT4245''', ignore_errors=True)
        self.query('CREATE SCHEMA dynamic_output')
        self.query('CREATE TABLE small(x VARCHAR(2000), y DOUBLE)')
        self.query('''INSERT INTO small VALUES ('Some string ... and some more', 2.2)''')
        self.query('create table groupt(id int, n double, v varchar(999))')
        self.query('''insert into groupt values (1,1,'aa'),
                                                (1,2,'ab'),
                                                (2,2,'ba')
                                                ''')
        self.query('create table target (a int, b double, c varchar(100));')
        self.query('''create connection SPOT4245 to 'a' user 'b' identified by 'c' ''')



class DynamicOutputCreateScript(Test):

    @requires('VAREMIT_SIMPLE_SET')
    def test_create_script_set(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='VAREMIT_SIMPLE_SET' and SCRIPT_TEXT LIKE 'CREATE % SET SCRIPT "VAREMIT_SIMPLE_SET" ("a" DOUBLE) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])

    @requires('VAREMIT_SIMPLE_SCALAR')
    def test_create_script_scalar(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='VAREMIT_SIMPLE_SCALAR' and SCRIPT_TEXT LIKE 'CREATE % SCALAR SCRIPT "VAREMIT_SIMPLE_SCALAR" ("a" DOUBLE) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])

    @requires('VAREMIT_SIMPLE_ALL_DYN')
    def test_create_script_all_dyn(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='VAREMIT_SIMPLE_ALL_DYN' and SCRIPT_TEXT LIKE 'CREATE % SCALAR SCRIPT "VAREMIT_SIMPLE_ALL_DYN" (...) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])

    @requires('VAREMIT_SIMPLE_SYNTAX_VAR')
    def test_create_script_syntax_var(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='VAREMIT_SIMPLE_SYNTAX_VAR' and SCRIPT_TEXT LIKE 'CREATE % SET SCRIPT "VAREMIT_SIMPLE_SYNTAX_VAR" (...) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])


class DynamicOutputTest(Test):

    @requires('VAREMIT_GENERIC_EMIT')
    def test_generic_emit(self):
        rows = self.query('''
            SELECT fn1.VAREMIT_GENERIC_EMIT('SUPERDYNAMIC') EMITS (a varchar(100));
            ''')
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])

    @requires('VAREMIT_ALL_GENERIC')
    def test_all_generic(self):
        rows = self.query('''
            SELECT fn1.VAREMIT_ALL_GENERIC('SUPERDYNAMIC') EMITS (a varchar(100));
            ''')
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])

    @requires('VAREMIT_GENERIC_EMIT')
    def test_correctness_emits_subquery(self):
        rows = self.query('''
            SELECT "A" || 'x' || "B" FROM (
            SELECT fn1.VAREMIT_GENERIC_EMIT('SUPERDYNAMIC') EMITS (a varchar(100), b varchar(100)));
            ''')
        self.assertRowEqual(('SUPERDYNAMICxSUPERDYNAMIC',), rows[0])

    @requires('VAREMIT_GENERIC_EMIT')
    def test_correctness_emits_with_grouping(self):
        rows = self.query('''
            SELECT 'X' || count(a) || 'X' FROM (
             SELECT fn1.VAREMIT_GENERIC_EMIT('SUPERDYNAMIC') EMITS (a varchar(100))
             FROM groupt GROUP BY id
            ) where a = 'SUPERDYNAMIC';
            ''')
        self.assertRowEqual(('X2X',), rows[0])


    @requires('VAREMIT_GENERIC_EMIT')
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


    @requires('VAREMIT_METADATA_SET_EMIT')
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
        self.assertTrue(rows[2][0] in stringType.get(udf.opts.lang))
        self.assertRowEqual(('VARCHAR(123) UTF8',1), rows[3])
        self.assertRowEqual(('123',1.0), rows[6])
        self.assertRowEqual(('B',1.0), rows[7])
        self.assertTrue(rows[8][0] in numType.get(udf.opts.lang))
        self.assertRowEqual(('DOUBLE',1.0), rows[9])


class DynamicOutputWrongUsage(Test):

    @requires('VAREMIT_GENERIC_EMIT')
    def test_error_emit_missing(self):
        #with self.assertRaisesRegex(Exception, 'The script has dynamic return arguments. Either specify the return arguments in the query via EMITS or implement the method (default_output_columns|getDefaultOutputColumns|defaultOutputColumns) in the UDF'):
        with self.assertRaisesRegex(Exception, 'The script has dynamic return arguments. Either specify the return arguments in the query via EMITS or implement the method'):
            self.query('''SELECT fn1.VAREMIT_GENERIC_EMIT(1)''')

    @requires('VAREMIT_GENERIC_EMIT')
    def test_error_empty_emit(self):
        with self.assertRaisesRegex(Exception, 'Empty return argument definition is not allowed'):
            self.query('''SELECT fn1.VAREMIT_GENERIC_EMIT(1) EMITS ();''')

    @requires('VAREMIT_GENERIC_EMIT')
    def test_error_empty_emit_2(self):
        with self.assertRaisesRegex(Exception, 'syntax error'):
            self.query('''SELECT fn1.VAREMIT_GENERIC_EMIT(1) EMITS (a);''')
            
    @requires('VAREMIT_GENERIC_EMIT')
    def test_error_wrong_emit(self):
        with self.assertRaisesRegex(Exception, 'syntax error'):
            self.query('''SELECT fn1.VAREMIT_GENERIC_EMIT(1) EMITS (int);''')

    @requires('VAREMIT_GENERIC_EMIT')
    def test_error_redundant_name(self):
        with self.assertRaisesRegex(Exception, 'Return argument A is declared more than once'):
            self.query('''SELECT fn1.VAREMIT_GENERIC_EMIT(1) EMITS (a int, b int, a int);''')

    @requires('VAREMIT_NON_VAR_EMIT')
    def test_error_non_var_emit(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition. Dynamic return arguments are not supported in this case'):
            self.query('''SELECT fn1.VAREMIT_NON_VAR_EMIT(1) EMITS (a double);''')

    @requires('VAREMIT_NON_VAR_EMIT')
    def test_error_non_var_emit_2(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition. Dynamic return arguments are not supported in this case'):
            self.query('''SELECT fn1.VAREMIT_NON_VAR_EMIT(1) EMITS ();''')

    @requires('VAREMIT_SIMPLE_RETURNS')
    def test_error_returns_not_supported(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition. Dynamic return arguments are not supported in this case'):
            self.query('''select fn1.VAREMIT_SIMPLE_RETURNS(1) EMITS (a INT);''')

    def test_error_built_in_set_not_supported(self):
        with self.assertRaisesRegex(Exception, 'emits specification is not allowed for built-in functions'):
            self.query('''SELECT AVG(a) EMITS(a int) FROM VAREMITS;''')

    def test_error_built_in_scalar_not_supported(self):
        with self.assertRaisesRegex(Exception, 'emits specification is not allowed for built-in functions'):
            self.query('''SELECT -ABS(a) EMITS(a int) FROM VAREMITS;''')



class DynamicOutputInsertInto(Test):

    @requires('VAREMIT_EMIT_INPUT')
    def test_insert_basic(self):
        self.query('''delete from target;''')
        self.query('''
            insert into target select fn1.VAREMIT_EMIT_INPUT(1, CAST (1.1 AS DOUBLE), 'a');
            ''')
        rows = self.query('''
            select * from target;
            ''')
        self.assertRowEqual((1, 1.1, 'a'), rows[0])
        self.query('''delete from target;''')

    @requires('VAREMIT_EMIT_INPUT_WITH_META_CHECK')
    def test_insert_metadata_correctness(self):
        self.query('''
            insert into target select fn1.VAREMIT_EMIT_INPUT_WITH_META_CHECK(cast (2 as int), CAST (2.2 AS DOUBLE), cast ('b' as varchar(100)));
            ''')
        rows = self.query('''
            select * from target;
            ''')
        self.assertRowEqual((2, 2.2, 'b'), rows[0])
        self.query('''delete from target;''')

    @requires('VAREMIT_EMIT_INPUT')
    def test_insert_target_columns_change_order(self):
        self.query('''
            insert into target (c, b, a) select fn1.VAREMIT_EMIT_INPUT('c', CAST (3.3 AS DOUBLE), 3);
            ''')
        rows = self.query('''
            select * from target;
            ''')
        self.assertRowEqual((3, 3.3, 'c'), rows[0])
        self.query('''delete from target;''')

    @requires('VAREMIT_EMIT_INPUT')
    def test_insert_target_columns_subset(self):
        self.query('''
            insert into target (b) select fn1.VAREMIT_EMIT_INPUT(CAST (4.4 AS DOUBLE));
            ''')
        rows = self.query('''
            select * from target;
            ''')
        self.assertRowEqual((None, 4.4, None), rows[0])
        self.query('''delete from target;''')

    @requires('VAREMIT_EMIT_INPUT')
    def test_insert_emits_not_allowed(self):
        with self.assertRaisesRegex(Exception, 'The return arguments for EMITS functions are inferred from the table to insert into. Specification of EMITS is not allowed in this case.'):
            self.query('''insert into target select FN1.VAREMIT_EMIT_INPUT(1) emits (a int);''')


class DynamicOutputCreateTableAs(Test):

    @requires('VAREMIT_EMIT_INPUT')
    def test_insert_basic(self):
        self.query('''drop table if exists targetcreated;''')
        self.query('''
            create table targetcreated as select fn1.VAREMIT_EMIT_INPUT(1, CAST (1.1 AS DOUBLE), 'a') emits (a decimal(20,0), b double, c varchar(100));
            ''')
        rows = self.query('''
            select * from targetcreated;
            ''')
        self.assertRowEqual((1, 1.1, 'a'), rows[0])
        rows = self.query('''
            describe targetcreated;
            ''')
        self.assertRowEqual(('A', 'DECIMAL(20,0)'), rows[0][0:2])
        self.assertRowEqual(('B', 'DOUBLE'), rows[1][0:2])
        self.assertRowEqual(('C', 'VARCHAR(100) UTF8'), rows[2][0:2])

## #####################################################
## The same as above but now with default output columns
## #####################################################

class DefaultDynamicOutputCreateScript(Test):

    @requires('DEFAULT_VAREMIT_SIMPLE_SET')
    def test_create_script_set(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='DEFAULT_VAREMIT_SIMPLE_SET' and SCRIPT_TEXT LIKE 'CREATE % SET SCRIPT "DEFAULT_VAREMIT_SIMPLE_SET" ("a" DOUBLE) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])

    @requires('DEFAULT_VAREMIT_SIMPLE_SCALAR')
    def test_create_script_scalar(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='DEFAULT_VAREMIT_SIMPLE_SCALAR' and SCRIPT_TEXT LIKE 'CREATE % SCALAR SCRIPT "DEFAULT_VAREMIT_SIMPLE_SCALAR" ("a" DOUBLE) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])

    @requires('DEFAULT_VAREMIT_SIMPLE_ALL_DYN')
    def test_create_script_all_dyn(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='DEFAULT_VAREMIT_SIMPLE_ALL_DYN' and SCRIPT_TEXT LIKE 'CREATE % SCALAR SCRIPT "DEFAULT_VAREMIT_SIMPLE_ALL_DYN" (...) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])

    @requires('DEFAULT_VAREMIT_SIMPLE_SYNTAX_VAR')
    def test_create_script_syntax_var(self):
        rows = self.query('''
            select count(*) from exa_all_scripts where script_name='DEFAULT_VAREMIT_SIMPLE_SYNTAX_VAR' and SCRIPT_TEXT LIKE 'CREATE % SET SCRIPT "DEFAULT_VAREMIT_SIMPLE_SYNTAX_VAR" (...) EMITS (...) AS%';
            ''')
        self.assertRowEqual((1,), rows[0])


class DefaultDynamicOutputTest(Test):

    @requires('DEFAULT_VAREMIT_GENERIC_EMIT')
    def test_generic_emit(self):
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT('SUPERDYNAMIC') EMITS (a varchar(100));
            ''')
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT('SUPERDYNAMIC');
            ''')
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])

    @requires('DEFAULT_VAREMIT_ALL_GENERIC')
    def test_all_generic(self):
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_ALL_GENERIC('SUPERDYNAMIC') EMITS (a varchar(100));
            ''')
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_ALL_GENERIC('SUPERDYNAMIC');
            ''')
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])


    @requires('DEFAULT_VAREMIT_GENERIC_EMIT')
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

    @requires('DEFAULT_VAREMIT_GENERIC_EMIT')
    def test_correctness_emits_with_grouping(self):
        rows = self.query('''
            SELECT 'X' || count(a) || 'X' FROM (
             SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT('SUPERDYNAMIC') EMITS (a varchar(100))
             FROM groupt GROUP BY id
            ) where a = 'SUPERDYNAMIC';
            ''')
        self.assertRowEqual(('X2X',), rows[0])
        rows = self.query('''
            SELECT 'X' || count(a) || 'X' FROM (
             SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT('SUPERDYNAMIC')
             FROM groupt GROUP BY id
            ) where a = 'SUPERDYNAMIC';
            ''')
        self.assertRowEqual(('X2X',), rows[0])


    @requires('DEFAULT_VAREMIT_GENERIC_EMIT')
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


    @requires('DEFAULT_VAREMIT_METADATA_SET_EMIT')
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
        self.assertTrue(rows[2][0] in stringType.get(udf.opts.lang))
        self.assertRowEqual(('VARCHAR(123) UTF8',1), rows[3])
        self.assertRowEqual(('123',1.0), rows[6])
        self.assertRowEqual(('B',1.0), rows[7])
        self.assertTrue(rows[8][0] in numType.get(udf.opts.lang))
        self.assertRowEqual(('DOUBLE',1.0), rows[9])
        # now again with intrinsic emits clause
        rows = self.query('''
            SELECT fn1.DEFAULT_VAREMIT_METADATA_SET_EMIT(1) 
            FROM DUAL
            ''')
        self.assertRowEqual(('2',1.0), rows[0])
        self.assertRowEqual(('A',1.0), rows[1])
        self.assertTrue(rows[2][0] in stringType.get(udf.opts.lang))
        self.assertRowEqual(('VARCHAR(123) UTF8',1), rows[3])
        self.assertRowEqual(('123',1.0), rows[6])
        self.assertRowEqual(('B',1.0), rows[7])
        self.assertTrue(rows[8][0] in numType.get(udf.opts.lang))
        self.assertRowEqual(('DOUBLE',1.0), rows[9])


class DefaultDynamicOutputWrongUsage(Test):

    ## @requires('DEFAULT_VAREMIT_GENERIC_EMIT')
    ## def test_error_emit_missing(self):
    ##     with self.assertRaisesRegex(Exception, 'The script has dynamic return args, but EMITS specification is missing in the query'):
    ##         self.query('''SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(1)''')

    @requires('DEFAULT_VAREMIT_GENERIC_EMIT')
    def test_error_empty_emit(self):
        with self.assertRaisesRegex(Exception, 'Empty return argument definition is not allowed'):
            self.query('''SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(1) EMITS ();''')

    @requires('DEFAULT_VAREMIT_GENERIC_EMIT')
    def test_error_empty_emit_2(self):
        with self.assertRaisesRegex(Exception, 'syntax error'):
            self.query('''SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(1) EMITS (a);''')
            
    @requires('DEFAULT_VAREMIT_GENERIC_EMIT')
    def test_error_wrong_emit(self):
        with self.assertRaisesRegex(Exception, 'syntax error'):
            self.query('''SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(1) EMITS (int);''')

    @requires('DEFAULT_VAREMIT_GENERIC_EMIT')
    def test_error_redundant_name(self):
        with self.assertRaisesRegex(Exception, 'Return argument A is declared more than once'):
            self.query('''SELECT fn1.DEFAULT_VAREMIT_GENERIC_EMIT(1) EMITS (a int, b int, a int);''')

    @requires('DEFAULT_VAREMIT_NON_VAR_EMIT')
    def test_error_non_var_emit(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition. Dynamic return arguments are not supported in this case'):
            self.query('''SELECT fn1.DEFAULT_VAREMIT_NON_VAR_EMIT(1) EMITS (a double);''')

    @requires('DEFAULT_VAREMIT_NON_VAR_EMIT')
    def test_error_non_var_emit_2(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition. Dynamic return arguments are not supported in this case'):
            self.query('''SELECT fn1.DEFAULT_VAREMIT_NON_VAR_EMIT(1) EMITS ();''')

    @requires('DEFAULT_VAREMIT_SIMPLE_RETURNS')
    def test_error_returns_not_supported(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition. Dynamic return arguments are not supported in this case'):
            self.query('''select fn1.DEFAULT_VAREMIT_SIMPLE_RETURNS(1) EMITS (a INT);''')




class DefaultDynamicOutputInsertInto(Test):

    @requires('DEFAULT_VAREMIT_EMIT_INPUT')
    def test_insert_basic(self):
        self.query('''delete from target;''')
        self.query('''
            insert into target select fn1.DEFAULT_VAREMIT_EMIT_INPUT(1, CAST (1.1 AS DOUBLE), 'a');
            ''')
        rows = self.query('''
            select * from target;
            ''')
        self.assertRowEqual((1, 1.1, 'a'), rows[0])
        self.query('''delete from target;''')

    @requires('DEFAULT_VAREMIT_EMIT_INPUT_WITH_META_CHECK')
    def test_insert_metadata_correctness(self):
        self.query('''
            insert into target select fn1.DEFAULT_VAREMIT_EMIT_INPUT_WITH_META_CHECK(cast (2 as int), CAST (2.2 AS DOUBLE), cast ('b' as varchar(100)));
            ''')
        rows = self.query('''
            select * from target;
            ''')
        self.assertRowEqual((2, 2.2, 'b'), rows[0])
        self.query('''delete from target;''')

    @requires('DEFAULT_VAREMIT_EMIT_INPUT')
    def test_insert_target_columns_change_order(self):
        self.query('''
            insert into target (c, b, a) select fn1.DEFAULT_VAREMIT_EMIT_INPUT('c', CAST (3.3 AS DOUBLE), 3);
            ''')
        rows = self.query('''
            select * from target;
            ''')
        self.assertRowEqual((3, 3.3, 'c'), rows[0])
        self.query('''delete from target;''')

    @requires('DEFAULT_VAREMIT_EMIT_INPUT')
    def test_insert_target_columns_subset(self):
        self.query('''
            insert into target (b) select fn1.DEFAULT_VAREMIT_EMIT_INPUT(CAST (4.4 AS DOUBLE));
            ''')
        rows = self.query('''
            select * from target;
            ''')
        self.assertRowEqual((None, 4.4, None), rows[0])
        self.query('''delete from target;''')

    @requires('DEFAULT_VAREMIT_EMIT_INPUT')
    def test_insert_emits_not_allowed(self):
        with self.assertRaisesRegex(Exception, 'The return arguments for EMITS functions are inferred from the table to insert into. Specification of EMITS is not allowed in this case.'):
            self.query('''insert into target select FN1.DEFAULT_VAREMIT_EMIT_INPUT(1) emits (a int);''')


class DefaultDynamicOutputCreateTableAs(Test):

    @requires('DEFAULT_VAREMIT_EMIT_INPUT')
    def test_insert_basic(self):
        self.query('''drop table if exists targetcreated;''')
        self.query('''
            create table targetcreated as select fn1.DEFAULT_VAREMIT_EMIT_INPUT(1, CAST (1.1 AS DOUBLE), 'a') emits (a decimal(20,0), b double, c varchar(100));
            ''')
        rows = self.query('''
            select * from targetcreated;
            ''')
        self.assertRowEqual((1, 1.1, 'a'), rows[0])
        rows = self.query('''
            describe targetcreated;
            ''')
        self.assertRowEqual(('A', 'DECIMAL(20,0)'), rows[0][0:2])
        self.assertRowEqual(('B', 'DOUBLE'), rows[1][0:2])
        self.assertRowEqual(('C', 'VARCHAR(100) UTF8'), rows[2][0:2])


class DefaultDynamicOutputEmptyStringResult(Test):

    @requires('DEFAULT_VAREMIT_EMPTY_DEF')
    def test_empty_string_error(self):
        with self.assertRaisesRegex(Exception, 'Empty default output columns'):
            self.query('''select fn1.DEFAULT_VAREMIT_EMPTY_DEF(42.42);''')
        rows = self.query('''select fn1.DEFAULT_VAREMIT_EMPTY_DEF(42.42) emits (x double);''')
        self.assertRowEqual((1.4,), rows[0])



class DefaultDynamicOutputFromInputMeta(Test):

    @requires('COPY_RELATION')
    def test_copy_relation(self):
        rows = self.query('''
            select fn1.copy_relation(1,2,3);
            ''')
        self.assertRowEqual((1, 2, 3), rows[0])




class DynamicOutFromConnectionsAndViews(Test):

    def getConnection(self, username, password):
        client = exatest.ODBCClient('exatest')
        self.log.debug('connecting to DSN "exa" for user {username}'.format(username=username))
        client.connect(uid = username, pwd = password)
        return client

    def createUser(self, username, password):
        self.query('DROP USER IF EXISTS {username} CASCADE'.format(username = username))
        self.query('CREATE USER {username} IDENTIFIED BY "{password}"'.format(username = username, password = password))
        self.query('GRANT CREATE SESSION TO {username}'.format(username=username))


    def checkColumnNamesOfQuery(self,query,expected_rows):
        self.query('''drop table if exists targetcreated''')
        self.query('''create schema spot4245_tmp''')
        self.query('''create table targetcreated as ''' + str(query))
        rows = self.query('''describe targetcreated''')
        for i in range(len(expected_rows)):
            self.assertRowEqual(expected_rows[i][0:2], rows[i][0:2])
        self.query('''drop schema spot4245_tmp cascade''')
        #('A', 'DECIMAL(20,0)', 'TRUE', 'FALSE'), rows[0])
        #self.assertRowEqual(('B', 'DOUBLE', 'TRUE', 'FALSE'), rows[1])
        #self.assertRowEqual(('C', 'VARCHAR(100) UTF8', 'TRUE', 'FALSE'), rows[2])


    
    @requires('OUTPUT_COLUMNS_AS_IN_CONNECTION_SPOT4245')
    def test_dynamic_out_from_connection_SPOT4245(self):
        expected_rows = [('PASSWORD', 'DOUBLE', 'TRUE', 'FALSE'), ('A', 'DOUBLE', 'TRUE', 'FALSE'), ('B', 'DOUBLE', 'TRUE', 'FALSE'), ('C', 'DOUBLE', 'TRUE', 'FALSE')]
        self.checkColumnNamesOfQuery('''select fn1.OUTPUT_COLUMNS_AS_IN_CONNECTION_SPOT4245(1.0)''', expected_rows)

    @requires('OUTPUT_COLUMNS_AS_IN_CONNECTION_SPOT4245')
    def test_dynamic_out_from_connection_SPOT4245_fails_for_user_foo(self):
        self.createUser("foo","foo")
        self.query('grant execute on fn1.OUTPUT_COLUMNS_AS_IN_CONNECTION_SPOT4245 to foo')
        self.commit()
        foo_conn = self.getConnection('foo','foo')
        expected_rows = [('PASSWORD', 'DOUBLE', 'TRUE', 'FALSE'), ('A', 'DOUBLE', 'TRUE', 'FALSE'), ('B', 'DOUBLE', 'TRUE', 'FALSE'), ('C', 'DOUBLE', 'TRUE', 'FALSE')]
        with self.assertRaisesRegex(Exception, 'insufficient privileges for using connection SPOT4245 in script OUTPUT_COLUMNS_AS_IN_CONNECTION_SPOT4245'):
            foo_conn.query('''select fn1.OUTPUT_COLUMNS_AS_IN_CONNECTION_SPOT4245(1.0)''')
            self.assertEqual(["PASSWORD","A","B","C"], foo_conn.columnNames())
        self.query("drop user foo cascade")


    @requires('OUTPUT_COLUMNS_AS_IN_CONNECTION_SPOT4245')
    def test_dynamic_out_from_connection_SPOT4245_for_user_foo_with_view(self):
        self.query('create view fn1.OUTPUT_COLUMNS_AS_IN_CONNECTION_SPOT4245_view as select fn1.OUTPUT_COLUMNS_AS_IN_CONNECTION_SPOT4245(1.0)')
        self.createUser("foo","foo")
        self.query('grant select on fn1.OUTPUT_COLUMNS_AS_IN_CONNECTION_SPOT4245_view to foo')
        self.commit()
        foo_conn = self.getConnection('foo','foo')
        foo_conn.query('''select * from fn1.OUTPUT_COLUMNS_AS_IN_CONNECTION_SPOT4245_view''')
        self.assertEqual(["PASSWORD","A","B","C"], [colDescription[0] for colDescription in foo_conn.cursorDescription()])
        self.query("drop user foo cascade")


    
## ##########################################################################
if __name__ == '__main__':
    udf.main()
