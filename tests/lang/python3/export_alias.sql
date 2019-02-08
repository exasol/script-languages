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

create or replace python3 scalar script expal_use_param_foo_bar(...) returns int as
def generate_sql_for_export_spec(export_spec):
    if (export_spec.parameters['FOO'] == 'bar' and
        export_spec.parameters['BAR'] == 'foo' and
        export_spec.connection_name is None and
        export_spec.connection is None and
        export_spec.has_truncate is False and
        export_spec.has_replace is False and
        export_spec.created_by is None and
        export_spec.source_column_names[0] == '"T"."A"' and
        export_spec.source_column_names[1] == '"T"."Z"'):
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
    else:
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
/

create or replace python3 scalar script expal_use_connection_name(...) returns int as
def generate_sql_for_export_spec(export_spec):
    if (export_spec.parameters['FOO'] == 'bar' and
        export_spec.parameters['BAR'] == 'foo' and
        export_spec.connection_name == 'FOOCONN' and
        export_spec.connection is None and
        export_spec.has_truncate is False and
        export_spec.has_replace is False and
        export_spec.created_by is None and
        export_spec.source_column_names[0] == '"T"."A"' and
        export_spec.source_column_names[1] == '"T"."Z"'):
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
    else:
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
/

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
        export_spec.source_column_names[0] == '"T"."A"' and
        export_spec.source_column_names[1] == '"T"."Z"'):
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
    else:
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
/

create or replace python3 scalar script expal_use_has_truncate(...) returns int as
def generate_sql_for_export_spec(export_spec):
    if (export_spec.parameters['FOO'] == 'bar' and
        export_spec.parameters['BAR'] == 'foo' and
        export_spec.connection_name is None and
        export_spec.connection is None and
        export_spec.has_truncate is True and
        export_spec.has_replace is False and
        export_spec.created_by is None and
        export_spec.source_column_names[0] == '"T"."A"' and
        export_spec.source_column_names[1] == '"T"."Z"'):
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
    else:
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
/

create or replace python3 scalar script expal_use_replace_created_by(...) returns int as
def generate_sql_for_export_spec(export_spec):
    if (export_spec.parameters['FOO'] == 'bar' and
        export_spec.parameters['BAR'] == 'foo' and
        export_spec.connection_name is None and
        export_spec.connection is None and
        export_spec.has_truncate is False and
        export_spec.has_replace is True and
        export_spec.created_by == 'create table t(a int, z varchar(3000))' and
        export_spec.source_column_names[0] == '"T"."A"' and
        export_spec.source_column_names[1] == '"T"."Z"'):
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
    else:
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
/

create or replace python3 scalar script expal_use_column_name_lower_case(...) returns int as
def generate_sql_for_export_spec(export_spec):
    if (export_spec.parameters['FOO'] == 'bar' and
        export_spec.parameters['BAR'] == 'foo' and
        export_spec.connection_name is None and
        export_spec.connection is None and
        export_spec.has_truncate is False and
        export_spec.has_replace is False and
        export_spec.created_by is None and
        export_spec.source_column_names[0] == '"tl"."A"' and
        export_spec.source_column_names[1] == '"tl"."z"'):
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
    else:
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
/

create or replace python3 scalar script expal_use_column_selection(...) returns int as
def generate_sql_for_export_spec(export_spec):
    if (export_spec.parameters['FOO'] == 'bar' and
        export_spec.parameters['BAR'] == 'foo' and
        export_spec.connection_name is None and
        export_spec.connection is None and
        export_spec.has_truncate is False and
        export_spec.has_replace is False and
        export_spec.created_by is None and
        export_spec.source_column_names[0] == '"tl"."A"' and
        export_spec.source_column_names[1] == '"tl"."z"'):
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
    else:
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
/

create or replace python3 scalar script expal_use_query(...) returns int as
def generate_sql_for_export_spec(export_spec):
    if (export_spec.parameters['FOO'] == 'bar' and
        export_spec.parameters['BAR'] == 'foo' and
        export_spec.connection_name is None and
        export_spec.connection is None and
        export_spec.has_truncate is False and
        export_spec.has_replace is False and
        export_spec.created_by is None and
        export_spec.source_column_names[0] == '"col1"' and
        export_spec.source_column_names[1] == '"col2"'):
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('ok')"
    else:
        return "select " + exa.meta.script_schema + ".expal_test_pass_fail('failed')"
/
