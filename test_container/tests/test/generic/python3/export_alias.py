#!/usr/bin/env python3

from exasol_python_test_framework import udf


class ExportAliasTest(udf.TestCase):
    result_unknown = 0
    result_ok = 1
    result_failed = 2
    result_test_error = 3

    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('create or replace table fn2.t(a int, z varchar(3000))')
        self.query("insert into fn2.t values (1, 'x')")
        self.query('create or replace table fn2."tl"(a int, "z" varchar(3000))')
        self.query("insert into fn2.\"tl\" values (1, 'x')")
        self.query("create connection FOOCONN to 'a' user 'b' identified by 'c'", ignore_errors=True)
        
        # Create all EXPORT UDF scripts
        self.query(udf.fixindent('''
            create or replace python3 scalar script expal_test_pass_fail(res varchar(100)) emits (x int) as
            def run(ctx):
                result = ctx.res
                if result == "ok":
                    ctx.emit(1)
                elif result == "failed":
                    ctx.emit(2)
                else:
                    ctx.emit(3)
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 scalar script expal_use_param_foo_bar(...) returns int as
            def generate_sql_for_export_spec(export_spec):
                if (export_spec.parameters['FOO'] == 'bar' and
                    export_spec.parameters['BAR'] == 'foo' and
                    export_spec.connection_name is None and
                    export_spec.connection is None and
                    export_spec.has_truncate is False and
                    export_spec.has_replace is False and
                    export_spec.created_by is None and
                    export_spec.source_column_names[0] == '\\"T\\".\\"A\\"' and
                    export_spec.source_column_names[1] == '\\"T\\".\\"Z\\"'):
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
                else:
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 scalar script expal_use_connection_name(...) returns int as
            def generate_sql_for_export_spec(export_spec):
                if (export_spec.parameters['FOO'] == 'bar' and
                    export_spec.parameters['BAR'] == 'foo' and
                    export_spec.connection_name == 'FOOCONN' and
                    export_spec.connection is None and
                    export_spec.has_truncate is False and
                    export_spec.has_replace is False and
                    export_spec.created_by is None and
                    export_spec.source_column_names[0] == '\\"T\\".\\"A\\"' and
                    export_spec.source_column_names[1] == '\\"T\\".\\"Z\\"'):
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
                else:
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 scalar script expal_use_connection_info(...) returns int as
            def generate_sql_for_export_spec(export_spec):
                if (export_spec.parameters['FOO'] == 'bar' and
                    export_spec.parameters['BAR'] == 'foo' and
                    export_spec.connection_name is None and
                    export_spec.connection.type == 'password'and
                    export_spec.connection.address == 'a' and
                    export_spec.connection.user == 'b' and
                    export_spec.connection.password == 'c' and
                    export_spec.has_truncate is False and
                    export_spec.has_replace is False and
                    export_spec.created_by is None and
                    export_spec.source_column_names[0] == '\\"T\\".\\"A\\"' and
                    export_spec.source_column_names[1] == '\\"T\\".\\"Z\\"'):
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
                else:
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 scalar script expal_use_has_truncate(...) returns int as
            def generate_sql_for_export_spec(export_spec):
                if (export_spec.parameters['FOO'] == 'bar' and
                    export_spec.parameters['BAR'] == 'foo' and
                    export_spec.connection_name is None and
                    export_spec.connection is None and
                    export_spec.has_truncate is True and
                    export_spec.has_replace is False and
                    export_spec.created_by is None and
                    export_spec.source_column_names[0] == '\\"T\\".\\"A\\"' and
                    export_spec.source_column_names[1] == '\\"T\\".\\"Z\\"'):
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
                else:
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 scalar script expal_use_replace_created_by(...) returns int as
            def generate_sql_for_export_spec(export_spec):
                if (export_spec.parameters['FOO'] == 'bar' and
                    export_spec.parameters['BAR'] == 'foo' and
                    export_spec.connection_name is None and
                    export_spec.connection is None and
                    export_spec.has_truncate is False and
                    export_spec.has_replace is True and
                    export_spec.created_by == 'create table t(a int, z varchar(3000))' and
                    export_spec.source_column_names[0] == '\\"T\\".\\"A\\"' and
                    export_spec.source_column_names[1] == '\\"T\\".\\"Z\\"'):
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
                else:
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 scalar script expal_use_column_name_lower_case(...) returns int as
            def generate_sql_for_export_spec(export_spec):
                if (export_spec.parameters['FOO'] == 'bar' and
                    export_spec.parameters['BAR'] == 'foo' and
                    export_spec.connection_name is None and
                    export_spec.connection is None and
                    export_spec.has_truncate is False and
                    export_spec.has_replace is False and
                    export_spec.created_by is None and
                    export_spec.source_column_names[0] == '\\"tl\\".\\"A\\"' and
                    export_spec.source_column_names[1] == '\\"tl\\".\\"z\\"'):
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
                else:
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 scalar script expal_use_column_selection(...) returns int as
            def generate_sql_for_export_spec(export_spec):
                if (export_spec.parameters['FOO'] == 'bar' and
                    export_spec.parameters['BAR'] == 'foo' and
                    export_spec.connection_name is None and
                    export_spec.connection is None and
                    export_spec.has_truncate is False and
                    export_spec.has_replace is False and
                    export_spec.created_by is None and
                    export_spec.source_column_names[0] == '\\"tl\\".\\"A\\"' and
                    export_spec.source_column_names[1] == '\\"tl\\".\\"z\\"'):
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
                else:
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 scalar script expal_use_query(...) returns int as
            def generate_sql_for_export_spec(export_spec):
                if (export_spec.parameters['FOO'] == 'bar' and
                    export_spec.parameters['BAR'] == 'foo' and
                    export_spec.connection_name is None and
                    export_spec.connection is None and
                    export_spec.has_truncate is False and
                    export_spec.has_replace is False and
                    export_spec.created_by is None and
                    export_spec.source_column_names[0] == '\\"col1\\"' and
                    export_spec.source_column_names[1] == '\\"col2\\"'):
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
                else:
                    return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
            /
        '''))

    def test_export_use_params(self):
        rows = self.executeStatement("EXPORT fn2.t INTO SCRIPT fn1.expal_use_param_foo_bar with foo='bar' bar='foo'")
        self.assertEqual(self.result_ok, rows)

    def test_export_use_connection_name(self):
        rows = self.executeStatement("EXPORT fn2.t INTO SCRIPT fn1.expal_use_connection_name AT FOOCONN with foo='bar' bar='foo'")
        self.assertEqual(self.result_ok, rows)

    def test_export_use_connection_info(self):
        rows = self.executeStatement("EXPORT fn2.t INTO SCRIPT fn1.expal_use_connection_info AT 'a' USER 'b' IDENTIFIED BY 'c' with foo='bar' bar='foo'")
        self.assertEqual(self.result_ok, rows)

    def test_export_use_has_truncate(self):
        rows = self.executeStatement("EXPORT fn2.t INTO SCRIPT fn1.expal_use_has_truncate with foo='bar' bar='foo' truncate")
        self.assertEqual(self.result_ok, rows)

    def test_export_use_replace_created_by(self):
        rows = self.executeStatement("EXPORT fn2.t INTO SCRIPT fn1.expal_use_replace_created_by with foo='bar' bar='foo' replace created by 'create table t(a int, z varchar(3000))'")
        self.assertEqual(self.result_ok, rows)

    def test_export_use_column_name_lower_case(self):
        rows = self.executeStatement("EXPORT fn2.\"tl\" INTO SCRIPT fn1.expal_use_column_name_lower_case with foo='bar' bar='foo'")
        self.assertEqual(self.result_ok, rows)

    def test_export_use_column_selection(self):
        rows = self.executeStatement("EXPORT fn2.\"tl\"(a, \"z\") INTO SCRIPT fn1.expal_use_column_selection with foo='bar' bar='foo'")
        self.assertEqual(self.result_ok, rows)

    def test_export_use_query(self):
        rows = self.executeStatement("EXPORT (select a as 'col1', \"z\" as 'col2' from fn2.\"tl\") INTO SCRIPT fn1.expal_use_query with foo='bar' bar='foo'")
        self.assertEqual(self.result_ok, rows)


if __name__ == '__main__':
    udf.main()
