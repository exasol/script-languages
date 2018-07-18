create or replace lua scalar script expal_test_pass_fail(res varchar(100)) emits (x int) as
  function run(ctx)
    result = ctx.res
    if result == "ok" then
        ctx.emit(decimal(1))
    elseif result == "failed" then
        ctx.emit(decimal(2))
    else
        ctx.emit(decimal(3))
    end
  end
/

create or replace lua scalar script expal_use_param_foo_bar(...) returns varchar(2000) as
  function generate_sql_for_export_spec(export_spec)
    if export_spec.parameters.FOO == "bar" and
       export_spec.parameters.BAR == "foo" and
       export_spec.connection_name == nil and
       export_spec.connection == nil and
       export_spec.has_truncate == "false" and
       export_spec.has_replace == "false" and
       export_spec.created_by == nil and
       export_spec.source_column_names[1] == "\"T\".\"A\"" and
       export_spec.source_column_names[2] == "\"T\".\"Z\""
    then
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('ok')"
    else
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('failed')"
    end
  end
/

create or replace lua scalar script expal_use_connection_name(...) returns varchar(2000) as
  function generate_sql_for_export_spec(export_spec)
    if export_spec.parameters.FOO == "bar" and
       export_spec.parameters.BAR == "foo" and
       export_spec.connection_name == "FOOCONN" and
       export_spec.connection == nil and
       export_spec.has_truncate == "false" and
       export_spec.has_replace == "false" and
       export_spec.created_by == nil and
       export_spec.source_column_names[1] == "\"T\".\"A\"" and
       export_spec.source_column_names[2] == "\"T\".\"Z\""
    then
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('ok')"
    else
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('failed')"
    end
  end
/

create or replace lua scalar script expal_use_connection_info(...) returns varchar(2000) as
  function generate_sql_for_export_spec(export_spec)
    if export_spec.parameters.FOO == "bar" and
       export_spec.parameters.BAR == "foo" and
       export_spec.connection_name == nil and
       export_spec.connection.type == "password" and
       export_spec.connection.address == "a" and
       export_spec.connection.user == "b" and
       export_spec.connection.password == "c" and
       export_spec.has_truncate == "false" and
       export_spec.has_replace == "false" and
       export_spec.created_by == nil and
       export_spec.source_column_names[1] == "\"T\".\"A\"" and
       export_spec.source_column_names[2] == "\"T\".\"Z\""
    then
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('ok')"
    else
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('failed')"
    end
  end
/

create or replace lua scalar script expal_use_has_truncate(...) returns varchar(2000) as
  function generate_sql_for_export_spec(export_spec)
    if export_spec.parameters.FOO == "bar" and
       export_spec.parameters.BAR == "foo" and
       export_spec.connection_name == nil and
       export_spec.connection == nil and
       export_spec.has_truncate == "true" and
       export_spec.has_replace == "false" and
       export_spec.created_by == nil and
       export_spec.source_column_names[1] == "\"T\".\"A\"" and
       export_spec.source_column_names[2] == "\"T\".\"Z\""
    then
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('ok')"
    else
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('failed')"
    end
  end
/

create or replace lua scalar script expal_use_replace_created_by(...) returns varchar(2000) as
  function generate_sql_for_export_spec(export_spec)
    if export_spec.parameters.FOO == "bar" and
       export_spec.parameters.BAR == "foo" and
       export_spec.connection_name == nil and
       export_spec.connection == nil and
       export_spec.has_truncate == "false" and
       export_spec.has_replace == "true" and
       export_spec.created_by == "create table t(a int, z varchar(3000))" and
       export_spec.source_column_names[1] == "\"T\".\"A\"" and
       export_spec.source_column_names[2] == "\"T\".\"Z\""
    then
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('ok')"
    else
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('failed')"
    end
  end
/

create or replace lua scalar script expal_use_column_name_lower_case(...) returns varchar(2000) as
  function generate_sql_for_export_spec(export_spec)
    if export_spec.parameters.FOO == "bar" and
       export_spec.parameters.BAR == "foo" and
       export_spec.connection_name == nil and
       export_spec.connection == nil and
       export_spec.has_truncate == "false" and
       export_spec.has_replace == "false" and
       export_spec.created_by == nil and
       export_spec.source_column_names[1] == "\"tl\".\"A\"" and
       export_spec.source_column_names[2] == "\"tl\".\"z\""
    then
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('ok')"
    else
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('failed')"
    end
  end
/

create or replace lua scalar script expal_use_column_selection(...) returns varchar(2000) as
  function generate_sql_for_export_spec(export_spec)
    if export_spec.parameters.FOO == "bar" and
       export_spec.parameters.BAR == "foo" and
       export_spec.connection_name == nil and
       export_spec.connection == nil and
       export_spec.has_truncate == "false" and
       export_spec.has_replace == "false" and
       export_spec.created_by == nil and
       export_spec.source_column_names[1] == "\"tl\".\"A\"" and
       export_spec.source_column_names[2] == "\"tl\".\"z\""
    then
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('ok')"
    else
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('failed')"
    end
  end
/

create or replace lua scalar script expal_use_query(...) returns varchar(2000) as
  function generate_sql_for_export_spec(export_spec)
    if export_spec.parameters.FOO == "bar" and
       export_spec.parameters.BAR == "foo" and
       export_spec.connection_name == nil and
       export_spec.connection == nil and
       export_spec.has_truncate == "false" and
       export_spec.has_replace == "false" and
       export_spec.created_by == nil and
       export_spec.source_column_names[1] == "\"col1\"" and
       export_spec.source_column_names[2] == "\"col2\""
    then
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('ok')"
    else
        return "select " .. exa.meta.script_schema .. ".expal_test_pass_fail('failed')"
    end
  end
/
