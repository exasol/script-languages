create or replace r scalar script expal_test_pass_fail(res varchar(100)) emits (x int) as
run <- function(ctx)
{
    if (ctx$res == "ok") {
        ctx$emit(1)
    } else if (ctx$res == "failed") {
        ctx$emit(2)
    } else {
        ctx$emit(3)
    }
}
/

create or replace r scalar script expal_use_param_foo_bar(...) emits (x int) as
generate_sql_for_export_spec <- function(export_spec)
{
    if (export_spec$parameters$FOO == "bar" &&
        export_spec$parameters$BAR == "foo" &&
        is.null(export_spec$connection_name) &&
        is.null(export_spec$connection) &&
        !isTRUE(export_spec$has_truncate) &&
        !isTRUE(export_spec$has_replace) &&
        is.null(export_spec$created_by) &&
        export_spec$source_column_names[1] == "\"T\".\"A\"" &&
        export_spec$source_column_names[2] == "\"T\".\"Z\"") {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'ok'", ")", sep="")
    } else {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'failed'", ")", sep="")
    }
}
/

create or replace r scalar script expal_use_connection_name(...) emits (x int) as
generate_sql_for_export_spec <- function(export_spec)
{
    if (export_spec$parameters$FOO == "bar" &&
        export_spec$parameters$BAR == "foo" &&
        export_spec$connection_name == "FOOCONN" &&
        is.null(export_spec$connection) &&
        !isTRUE(export_spec$has_truncate) &&
        !isTRUE(export_spec$has_replace) &&
        is.null(export_spec$created_by) &&
        export_spec$source_column_names[1] == "\"T\".\"A\"" &&
        export_spec$source_column_names[2] == "\"T\".\"Z\"") {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'ok'", ")", sep="")
    } else {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'failed'", ")", sep="")
    }
}
/

create or replace r scalar script expal_use_connection_info(...) emits (x int) as
generate_sql_for_export_spec <- function(export_spec)
{
    if (export_spec$parameters$FOO == "bar" &&
        export_spec$parameters$BAR == "foo" &&
        is.null(export_spec$connection_name) &&
        export_spec$connection$type == "password" &&
        export_spec$connection$address == "a" &&
        export_spec$connection$user == "b" &&
        export_spec$connection$password == "c" &&
        !isTRUE(export_spec$has_truncate) &&
        !isTRUE(export_spec$has_replace) &&
        is.null(export_spec$created_by) &&
        export_spec$source_column_names[1] == "\"T\".\"A\"" &&
        export_spec$source_column_names[2] == "\"T\".\"Z\"") {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'ok'", ")", sep="")
    } else {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'failed'", ")", sep="")
    }
}
/

create or replace r scalar script expal_use_has_truncate(...) emits (x int) as
generate_sql_for_export_spec <- function(export_spec)
{
    if (export_spec$parameters$FOO == "bar" &&
        export_spec$parameters$BAR == "foo" &&
        is.null(export_spec$connection_name) &&
        is.null(export_spec$connection) &&
        isTRUE(export_spec$has_truncate) &&
        !isTRUE(export_spec$has_replace) &&
        is.null(export_spec$created_by) &&
        export_spec$source_column_names[1] == "\"T\".\"A\"" &&
        export_spec$source_column_names[2] == "\"T\".\"Z\"") {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'ok'", ")", sep="")
    } else {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'failed'", ")", sep="")
    }
}
/

create or replace r scalar script expal_use_replace_created_by(...) emits (x int) as
generate_sql_for_export_spec <- function(export_spec)
{
    if (export_spec$parameters$FOO == "bar" &&
        export_spec$parameters$BAR == "foo" &&
        is.null(export_spec$connection_name) &&
        is.null(export_spec$connection) &&
        !isTRUE(export_spec$has_truncate) &&
        isTRUE(export_spec$has_replace) &&
        export_spec$created_by == "create table t(a int, z varchar(3000))" &&
        export_spec$source_column_names[1] == "\"T\".\"A\"" &&
        export_spec$source_column_names[2] == "\"T\".\"Z\"") {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'ok'", ")", sep="")
    } else {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'failed'", ")", sep="")
    }
}
/

create or replace r scalar script expal_use_column_name_lower_case(...) emits (x int) as
generate_sql_for_export_spec <- function(export_spec)
{
    if (export_spec$parameters$FOO == "bar" &&
        export_spec$parameters$BAR == "foo" &&
        is.null(export_spec$connection_name) &&
        is.null(export_spec$connection) &&
        !isTRUE(export_spec$has_truncate) &&
        !isTRUE(export_spec$has_replace) &&
        is.null(export_spec$created_by) &&
        export_spec$source_column_names[1] == "\"tl\".\"A\"" &&
        export_spec$source_column_names[2] == "\"tl\".\"z\"") {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'ok'", ")", sep="")
    } else {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'failed'", ")", sep="")
    }
}
/

create or replace r scalar script expal_use_column_selection(...) emits (x int) as
generate_sql_for_export_spec <- function(export_spec)
{
    if (export_spec$parameters$FOO == "bar" &&
        export_spec$parameters$BAR == "foo" &&
        is.null(export_spec$connection_name) &&
        is.null(export_spec$connection) &&
        !isTRUE(export_spec$has_truncate) &&
        !isTRUE(export_spec$has_replace) &&
        is.null(export_spec$created_by) &&
        export_spec$source_column_names[1] == "\"tl\".\"A\"" &&
        export_spec$source_column_names[2] == "\"tl\".\"z\"") {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'ok'", ")", sep="")
    } else {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'failed'", ")", sep="")
    }
}
/

create or replace r scalar script expal_use_query(...) emits (x int) as
generate_sql_for_export_spec <- function(export_spec)
{
    if (export_spec$parameters$FOO == "bar" &&
        export_spec$parameters$BAR == "foo" &&
        is.null(export_spec$connection_name) &&
        is.null(export_spec$connection) &&
        !isTRUE(export_spec$has_truncate) &&
        !isTRUE(export_spec$has_replace) &&
        is.null(export_spec$created_by) &&
        export_spec$source_column_names[1] == "\"col1\"" &&
        export_spec$source_column_names[2] == "\"col2\"") {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'ok'", ")", sep="")
    } else {
        paste("select ", exa$meta$script_schema, ".", "expal_test_pass_fail(", "'failed'", ")", sep="")
    }
}
/
