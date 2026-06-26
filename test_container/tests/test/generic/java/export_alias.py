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
        self.query('create or replace table fn2.\"tl\"(a int, \"z\" varchar(3000))')
        self.query("insert into fn2.\"tl\" values (1, 'x')")
        self.query("create connection FOOCONN to 'a' user 'b' identified by 'c'", ignore_errors=True)
        
        self.query('OPEN SCHEMA FN1')
        
        # Create all EXPORT UDF scripts
        self.query(udf.fixindent('''
            create or replace java set script expal_test_pass_fail(res varchar(100)) emits (x int) as
            %jvmoption -Xms64m -Xmx128m -Xss512k;
            class EXPAL_TEST_PASS_FAIL {
              public static void run(ExaMetadata meta, ExaIterator iter) throws Exception {
                    String result = iter.getString("res");
                    if (result.equals("ok")) {
                        iter.emit(1);
                    }
                    else if (result.equals("failed")) {
                        iter.emit(2);
                    }
                    else {
                        iter.emit(3);
                    }
              }
            }
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace java set script expal_use_param_foo_bar(...) emits (x varchar(2000)) as
            %jvmoption -Xms64m -Xmx128m -Xss512k;
            class EXPAL_USE_PARAM_FOO_BAR {
              static String generateSqlForExportSpec(ExaMetadata exa, ExaExportSpecification exportSpecification) {
                    if (exportSpecification.getParameters().size() == 2 &&
                        exportSpecification.getParameters().get("FOO").equals("bar") &&
                        exportSpecification.getParameters().get("BAR").equals("foo") &&
                        !exportSpecification.hasConnectionName() &&
                        exportSpecification.getConnectionName() == null &&
                        !exportSpecification.hasConnectionInformation() &&
                        exportSpecification.getConnectionInformation() == null &&
                        !exportSpecification.hasTruncate() &&
                        !exportSpecification.hasReplace() &&
                        !exportSpecification.hasCreatedBy() &&
                        exportSpecification.getSourceColumnNames().size() == 2 &&
                        exportSpecification.getSourceColumnNames().get(0).equals("\\"T\\".\\"A\\"") &&
                        exportSpecification.getSourceColumnNames().get(1).equals("\\"T\\".\\"Z\\"")) {
                        return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
                    }
                    return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
              }
            }
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace java set script expal_use_connection_name(...) emits (x varchar(2000)) as
            %jvmoption -Xms64m -Xmx128m -Xss512k;
            class EXPAL_USE_CONNECTION_NAME {
              static String generateSqlForExportSpec(ExaMetadata exa, ExaExportSpecification exportSpecification) {
                  if (exportSpecification.getParameters().size() == 2 &&
                      exportSpecification.getParameters().get("FOO").equals("bar") &&
                      exportSpecification.getParameters().get("BAR").equals("foo") &&
                      exportSpecification.hasConnectionName() &&
                      exportSpecification.getConnectionName().equals("FOOCONN") &&
                      !exportSpecification.hasConnectionInformation() &&
                      exportSpecification.getConnectionInformation() == null &&
                      !exportSpecification.hasTruncate() &&
                      !exportSpecification.hasReplace() &&
                      !exportSpecification.hasCreatedBy() &&
                      exportSpecification.getSourceColumnNames().size() == 2 &&
                      exportSpecification.getSourceColumnNames().get(0).equals("\\"T\\".\\"A\\"") &&
                      exportSpecification.getSourceColumnNames().get(1).equals("\\"T\\".\\"Z\\"")) {
                      return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
                  }
                  return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
              }
            }
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace java set script expal_use_connection_info(...) emits (x varchar(2000)) as
            %jvmoption -Xms64m -Xmx128m -Xss512k;
            class EXPAL_USE_CONNECTION_INFO {
              static String generateSqlForExportSpec(ExaMetadata exa, ExaExportSpecification exportSpecification) {
                  if (exportSpecification.getParameters().size() == 2 &&
                      exportSpecification.getParameters().get("FOO").equals("bar") &&
                      exportSpecification.getParameters().get("BAR").equals("foo") &&
                      !exportSpecification.hasConnectionName() &&
                      exportSpecification.getConnectionName() == null &&
                      exportSpecification.hasConnectionInformation() &&
                      exportSpecification.getConnectionInformation().getAddress().equals("a") &&
                      exportSpecification.getConnectionInformation().getUser().equals("b") &&
                      exportSpecification.getConnectionInformation().getPassword().equals("c") &&
                      !exportSpecification.hasTruncate() &&
                      !exportSpecification.hasReplace() &&
                      !exportSpecification.hasCreatedBy() &&
                      exportSpecification.getSourceColumnNames().size() == 2 &&
                      exportSpecification.getSourceColumnNames().get(0).equals("\\"T\\".\\"A\\"") &&
                      exportSpecification.getSourceColumnNames().get(1).equals("\\"T\\".\\"Z\\"")) {
                      return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
                  }
                  return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
              }
            }
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace java set script expal_use_has_truncate(...) emits (x varchar(2000)) as
            %jvmoption -Xms64m -Xmx128m -Xss512k;
            class EXPAL_USE_HAS_TRUNCATE {
              static String generateSqlForExportSpec(ExaMetadata exa, ExaExportSpecification exportSpecification) {
                  if (exportSpecification.getParameters().size() == 2 &&
                      exportSpecification.getParameters().get("FOO").equals("bar") &&
                      exportSpecification.getParameters().get("BAR").equals("foo") &&
                      !exportSpecification.hasConnectionName() &&
                      exportSpecification.getConnectionName() == null &&
                      !exportSpecification.hasConnectionInformation() &&
                      exportSpecification.getConnectionInformation() == null &&
                      exportSpecification.hasTruncate() &&
                      !exportSpecification.hasReplace() &&
                      !exportSpecification.hasCreatedBy() &&
                      exportSpecification.getSourceColumnNames().size() == 2 &&
                      exportSpecification.getSourceColumnNames().get(0).equals("\\"T\\".\\"A\\"") &&
                      exportSpecification.getSourceColumnNames().get(1).equals("\\"T\\".\\"Z\\"")) {
                      return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
                  }
                  return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
              }
            }
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace java set script expal_use_replace_created_by(...) emits (x varchar(2000)) as
            %jvmoption -Xms64m -Xmx128m -Xss512k;
            class EXPAL_USE_REPLACE_CREATED_BY {
              static String generateSqlForExportSpec(ExaMetadata exa, ExaExportSpecification exportSpecification) {
                  if (exportSpecification.getParameters().size() == 2 &&
                      exportSpecification.getParameters().get("FOO").equals("bar") &&
                      exportSpecification.getParameters().get("BAR").equals("foo") &&
                      !exportSpecification.hasConnectionName() &&
                      exportSpecification.getConnectionName() == null &&
                      !exportSpecification.hasConnectionInformation() &&
                      exportSpecification.getConnectionInformation() == null &&
                      !exportSpecification.hasTruncate() &&
                      exportSpecification.hasReplace() &&
                      exportSpecification.hasCreatedBy() &&
                      exportSpecification.getCreatedBy().equals("create table t(a int, z varchar(3000))") &&
                      exportSpecification.getSourceColumnNames().size() == 2 &&
                      exportSpecification.getSourceColumnNames().get(0).equals("\\"T\\".\\"A\\"") &&
                      exportSpecification.getSourceColumnNames().get(1).equals("\\"T\\".\\"Z\\"")) {
                      return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
                  }
                  return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
              }
            }
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace java set script expal_use_column_name_lower_case(...) emits (x varchar(2000)) as
            %jvmoption -Xms64m -Xmx128m -Xss512k;
            class EXPAL_USE_COLUMN_NAME_LOWER_CASE {
              static String generateSqlForExportSpec(ExaMetadata exa, ExaExportSpecification exportSpecification) {
                    if (exportSpecification.getParameters().size() == 2 &&
                        exportSpecification.getParameters().get("FOO").equals("bar") &&
                        exportSpecification.getParameters().get("BAR").equals("foo") &&
                        !exportSpecification.hasConnectionName() &&
                        exportSpecification.getConnectionName() == null &&
                        !exportSpecification.hasConnectionInformation() &&
                        exportSpecification.getConnectionInformation() == null &&
                        !exportSpecification.hasTruncate() &&
                        !exportSpecification.hasReplace() &&
                        !exportSpecification.hasCreatedBy() &&
                        exportSpecification.getSourceColumnNames().size() == 2 &&
                        exportSpecification.getSourceColumnNames().get(0).equals("\\"tl\\".\\"A\\"") &&
                        exportSpecification.getSourceColumnNames().get(1).equals("\\"tl\\".\\"z\\"")) {
                        return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
                    }
                    return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
              }
            }
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace java set script expal_use_column_selection(...) emits (x varchar(2000)) as
            %jvmoption -Xms64m -Xmx128m -Xss512k;
            class EXPAL_USE_COLUMN_SELECTION {
              static String generateSqlForExportSpec(ExaMetadata exa, ExaExportSpecification exportSpecification) {
                    if (exportSpecification.getParameters().size() == 2 &&
                        exportSpecification.getParameters().get("FOO").equals("bar") &&
                        exportSpecification.getParameters().get("BAR").equals("foo") &&
                        !exportSpecification.hasConnectionName() &&
                        exportSpecification.getConnectionName() == null &&
                        !exportSpecification.hasConnectionInformation() &&
                        exportSpecification.getConnectionInformation() == null &&
                        !exportSpecification.hasTruncate() &&
                        !exportSpecification.hasReplace() &&
                        !exportSpecification.hasCreatedBy() &&
                        exportSpecification.getSourceColumnNames().size() == 2 &&
                        exportSpecification.getSourceColumnNames().get(0).equals("\\"tl\\".\\"A\\"") &&
                        exportSpecification.getSourceColumnNames().get(1).equals("\\"tl\\".\\"z\\"")) {
                        return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
                    }
                    return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
              }
            }
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace java set script expal_use_query(...) emits (x varchar(2000)) as
            %jvmoption -Xms64m -Xmx128m -Xss512k;
            class EXPAL_USE_QUERY {
              static String generateSqlForExportSpec(ExaMetadata exa, ExaExportSpecification exportSpecification) {
                    if (exportSpecification.getParameters().size() == 2 &&
                        exportSpecification.getParameters().get("FOO").equals("bar") &&
                        exportSpecification.getParameters().get("BAR").equals("foo") &&
                        !exportSpecification.hasConnectionName() &&
                        exportSpecification.getConnectionName() == null &&
                        !exportSpecification.hasConnectionInformation() &&
                        exportSpecification.getConnectionInformation() == null &&
                        !exportSpecification.hasTruncate() &&
                        !exportSpecification.hasReplace() &&
                        !exportSpecification.hasCreatedBy() &&
                        exportSpecification.getSourceColumnNames().size() == 2 &&
                        exportSpecification.getSourceColumnNames().get(0).equals("\\"col1\\"") &&
                        exportSpecification.getSourceColumnNames().get(1).equals("\\"col2\\"")) {
                        return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
                    }
                    return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
              }
            }
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
