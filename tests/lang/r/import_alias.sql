create or replace r set script impal_use_is_subselect(...) emits (x varchar(2000)) as
generate_sql_for_import_spec <- function(import_spec)
{
    if (isTRUE(import_spec$is_subselect)) {
       return (paste("select", " True"))
    } else
    {
      return (paste("select", " False"))
    }
}
/

create or replace r set script impal_use_connection_name(...) emits (x varchar(2000)) as
generate_sql_for_import_spec <- function(import_spec)
{
    paste("select '", import_spec$connection_name, "'",  sep="")
}
/

create or replace r scalar script impal_use_param_foo_bar(...) returns varchar(2000) as
generate_sql_for_import_spec <- function(import_spec)
{
    paste("select '", import_spec$parameters$FOO, "', '", import_spec$parameters$BAR, "'", sep="")
}
/

create or replace r set script impal_use_connection(...) emits (x varchar(2000)) as
generate_sql_for_import_spec <- function(import_spec)
{
    paste("select '", import_spec$connection$user, import_spec$connection$password, import_spec$connection$address, import_spec$connection$type, "'", sep="")
}
/

create or replace r set script impal_use_all(...) emits (x varchar(2000)) as
generate_sql_for_import_spec <- function(import_spec)
{
        is_sub = "FALSE"
    connection_string = 'X'
    connection_name = 'Y'
    foo = 'Z'
    types = 'T'
    names = 'N'
        if (import_spec$is_subselect) {
       is_sub = "TRUE";
    }
    if (!is.null(import_spec$connection) ) {
                connection_string = paste(import_spec$connection$user, import_spec$connection$password, import_spec$connection$address, import_spec$connection$type, sep="");
        }
        if (!is.null(import_spec$connection_name) ) {
                connection_name = import_spec$connection_name;
        }
        if (!is.null(import_spec$parameters) ) {
                foo = import_spec$parameters$FOO
        }
        if (!is.null(import_spec$subselect_column_types) ) {
                for (type in import_spec$subselect_column_types)
                {
                types = paste(types, type, sep="");
                }
                for (name in import_spec$subselect_column_names)
                {
                names = paste(names, name, sep="");
                }
        }
        return (paste("select 1, '" , is_sub , '_' , connection_name , '_' , connection_string , '_' ,  foo , '_' , types , '_' , names , "'", sep=""));
}
/
