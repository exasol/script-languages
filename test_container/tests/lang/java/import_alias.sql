create or replace java set script impal_use_is_subselect(...) emits (x varchar(2000)) as
%jvmoption -Xms64m -Xmx128m -Xss512k;
class IMPAL_USE_IS_SUBSELECT {
  static String generateSqlForImportSpec(ExaMetadata exa, ExaImportSpecification importSpecification) {
    return "select " + importSpecification.isSubselect();
  }
}
/

create or replace java set script impal_use_param_foo_bar(...) emits (x varchar(2000)) as
%jvmoption -Xms64m -Xmx128m -Xss512k;
class IMPAL_USE_PARAM_FOO_BAR {
  static String generateSqlForImportSpec(ExaMetadata exa, ExaImportSpecification importSpecification) {
        return "select '" +  importSpecification.getParameters().get("FOO") + "', '" + importSpecification.getParameters().get("BAR") + "'";
  }
}
/

create or replace java set script impal_use_connection_name(...) emits (x varchar(2000)) as
%jvmoption -Xms64m -Xmx128m -Xss512k;
class IMPAL_USE_CONNECTION_NAME {
  static String generateSqlForImportSpec(ExaMetadata exa, ExaImportSpecification importSpecification) {
        return "select '" +  importSpecification.getConnectionName() + "'";
  }
}
/

create or replace java set script impal_use_connection(...) emits (x varchar(2000)) as
%jvmoption -Xms64m -Xmx128m -Xss512k;
class IMPAL_USE_CONNECTION {
  static String generateSqlForImportSpec(ExaMetadata exa, ExaImportSpecification importSpecification) {
        ExaConnectionInformation conn =  importSpecification.getConnectionInformation();
        return "select '" +  conn.getUser() +  conn.getPassword() +  conn.getAddress() +  conn.getType().toString().toLowerCase() + "'";
  }
}
/

create or replace java set script impal_use_all(...) emits (x varchar(2000)) as
%jvmoption -Xms64m -Xmx128m -Xss512k;
class IMPAL_USE_ALL {
  static String generateSqlForImportSpec(ExaMetadata exa, ExaImportSpecification importSpecification) {
            String is_sub = "FALSE";
        if (importSpecification.isSubselect()) {
                        is_sub = "TRUE";
                }
        String connection_string = "X";
        String connection_name = "Y";
        String foo = "Z";
        String types = "T";
        String names = "N";
        if (importSpecification.hasConnectionInformation()) {
                ExaConnectionInformation conn =  importSpecification.getConnectionInformation();
                connection_string = conn.getUser() +  conn.getPassword() +  conn.getAddress() +  conn.getType().toString().toLowerCase();
        }
        if (importSpecification.hasConnectionName()) {
                connection_name = importSpecification.getConnectionName();
        }
        if (importSpecification.getParameters().get("FOO") != null) {
                foo = importSpecification.getParameters().get("FOO");
        }
        if (importSpecification.getSubselectColumnNames().size() > 0) {
                for (int i = 0; i < importSpecification.getSubselectColumnNames().size(); i++) {
                        types = types + importSpecification.getSubselectColumnSqlTypes().get(i);
                        names = names + importSpecification.getSubselectColumnNames().get(i);
                }
        }
        return "select 1, '" + is_sub + '_' + connection_name + '_' + connection_string + '_' +  foo + '_' + types + '_' + names + "'";
  }
}
/
