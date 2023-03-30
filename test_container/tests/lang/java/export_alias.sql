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
            exportSpecification.getSourceColumnNames().get(0).equals("\"T\".\"A\"") &&
            exportSpecification.getSourceColumnNames().get(1).equals("\"T\".\"Z\"")) {
            return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
        }
        return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
  }
}
/

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
          exportSpecification.getSourceColumnNames().get(0).equals("\"T\".\"A\"") &&
          exportSpecification.getSourceColumnNames().get(1).equals("\"T\".\"Z\"")) {
          return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
      }
      return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
  }
}
/

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
          exportSpecification.getSourceColumnNames().get(0).equals("\"T\".\"A\"") &&
          exportSpecification.getSourceColumnNames().get(1).equals("\"T\".\"Z\"")) {
          return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
      }
      return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
  }
}
/

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
          exportSpecification.getSourceColumnNames().get(0).equals("\"T\".\"A\"") &&
          exportSpecification.getSourceColumnNames().get(1).equals("\"T\".\"Z\"")) {
          return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
      }
      return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
  }
}
/

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
          exportSpecification.getSourceColumnNames().get(0).equals("\"T\".\"A\"") &&
          exportSpecification.getSourceColumnNames().get(1).equals("\"T\".\"Z\"")) {
          return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
      }
      return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
  }
}
/

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
            exportSpecification.getSourceColumnNames().get(0).equals("\"tl\".\"A\"") &&
            exportSpecification.getSourceColumnNames().get(1).equals("\"tl\".\"z\"")) {
            return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
        }
        return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
  }
}
/

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
            exportSpecification.getSourceColumnNames().get(0).equals("\"tl\".\"A\"") &&
            exportSpecification.getSourceColumnNames().get(1).equals("\"tl\".\"z\"")) {
            return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
        }
        return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
  }
}
/

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
            exportSpecification.getSourceColumnNames().get(0).equals("\"col1\"") &&
            exportSpecification.getSourceColumnNames().get(1).equals("\"col2\"")) {
            return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('ok')";
        }
        return "select " + exa.getScriptSchema() + ".expal_test_pass_fail('failed')";
  }
}
/
