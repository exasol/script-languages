CREATE R SCALAR SCRIPT
get_database_name() returns varchar(300) AS
run <- function(ctx)
	exa$meta$database_name
/
create R scalar script
get_database_version() returns varchar(20) as
run <- function(ctx)
	exa$meta$database_version
/
create R scalar script
get_script_language() emits (s1 varchar(300), s2 varchar(300)) as
run <- function(ctx)
	ctx$emit(exa$meta$script_language, "R")
/
create R scalar script
get_script_name() returns varchar(200) as
run <- function(ctx)
	exa$meta$script_name
/
create R scalar script
get_script_schema() returns varchar(200) as
run <- function(ctx)
        exa$meta$script_schema
/
create R scalar script
get_current_user() returns varchar(200) as
run <- function(ctx)
        exa$meta$current_user
/
create R scalar script
get_scope_user() returns varchar(200) as
run <- function(ctx)
        exa$meta$scope_user
/
create R scalar script
get_current_schema() returns varchar(200) as
run <- function(ctx)
        exa$meta$current_schema
/
create R scalar script
get_script_code() returns varchar(2000) as
run <- function(ctx)
	exa$meta$script_code
/
create R scalar script
get_session_id() returns varchar(200) as
run <- function(ctx)
	exa$meta$session_id
/
create R scalar script
get_statement_id() returns number as
run <- function(ctx)
	exa$meta$statement_id
/
create R scalar script
get_node_count() returns number as
run <- function(ctx)
	exa$meta$node_count
/
create R scalar script
get_node_id() returns number as
run <- function(ctx)
	exa$meta$node_id
/
create R scalar script
get_vm_id() returns varchar(200) as
run <- function(ctx)
 	exa$meta$vm_id
/
create R scalar script
get_input_type_scalar() returns varchar(200) as
run <- function(ctx)
 	exa$meta$input_type
/
create R set script
get_input_type_set(a double) returns varchar(200) as
run <- function(ctx)
 	exa$meta$input_type
/
create R scalar script
get_input_column_count_scalar(c1 double, c2 varchar(100))
returns number as
run <- function(ctx)
 	exa$meta$input_column_count
/
create R set script
get_input_column_count_set(c1 double, c2 varchar(100))
returns number as
run <- function(ctx)
     exa$meta$input_column_count
/
create R scalar script
get_input_columns(c1 double, c2 varchar(200))
emits (column_id number, column_name varchar(200), column_type varchar(20),
 	   column_sql_type varchar(20), column_precision number, column_scale number,
  	   column_length number) as
run <- function(ctx) {
	cols <- exa$meta$input_columns
  	for (i in 1:length(cols)) {
 		name  <- cols[[i]]$name
 		precision <- cols[[i]]$precision
 		thetype <- cols[[i]]$type
 		sql_type <- cols[[i]]$sql_type
 		scale <- cols[[i]]$scale
 		length <- cols[[i]]$length
 		if (is.null(name) || is.na(name)) name <- 'no-name'
 		if (is.null(thetype) || is.na(thetype)) thetype <- 'no-type'
 		if (is.null(sql_type) || is.na(sql_type)) sql_type <- 'no-sql-type'
 		if (is.null(precision) || is.na(precision)) precision <- 0
 		if (is.null(scale) || is.na(scale)) scale <- 0
 		if (is.null(length) || is.na(length)) length <- 0
 		ctx$emit(i, name, thetype, sql_type, precision, scale, length)
	}
}
/
create R scalar script
get_output_type_return()
returns varchar(200) as
run <- function(ctx)
 	exa$meta$output_type
/
create R scalar script
get_output_type_emit()
emits (t varchar(200)) as
run <- function(ctx)
   ctx$emit(exa$meta$output_type)
/
create R scalar script
get_output_column_count_return()
returns number as
run <- function(ctx)
 	exa$meta$output_column_count
/
create R scalar script
get_output_column_count_emit()
emits (x number, y number, z number) as
run <- function(ctx)
	ctx$emit(exa$meta$output_column_count,exa$meta$output_column_count,exa$meta$output_column_count)
/
create R scalar script
get_output_columns()
emits (column_id number, column_name varchar(200), column_type varchar(20),
 	   column_sql_type varchar(20), column_precision number, column_scale number,
  	   column_length number) as
run <- function(ctx) {
	cols <- exa$meta$output_columns
        for (i in 1:length(cols)) {
        name  <- cols[[i]]$name
        precision <- cols[[i]]$precision
        thetype <- cols[[i]]$type
        sql_type <- cols[[i]]$sql_type
        scale <- cols[[i]]$scale
        length <- cols[[i]]$length
        if (is.null(name) || is.na(name)) name <- 'no-name'
        if (is.null(thetype) || is.na(thetype)) thetype <- 'no-type'
        if (is.null(sql_type) || is.na(sql_type)) sql_type <- 'no-sql-type'
        if (is.null(precision) || is.na(precision)) precision <- 0
        if (is.null(scale) || is.na(scale)) scale <- 0
        if (is.null(length) || is.na(length)) length <- 0
        ctx$emit(i, name, thetype, sql_type, precision, scale, length)
    }
}
/

create R scalar script
get_precision_scale_length(n decimal(6,3), v varchar(10))
emits (precision1 number, scale1 number, length1 number, precision2 number, scale2 number, length2 number) as
run <- function(ctx) {
    v <- exa$meta$input_columns[[1]]
    precision1 <- v$precision
    scale1 <- v$scale
    length1 <- v$length
    w <- exa$meta$input_columns[[2]]
    precision2 <- w$precision
    scale2 <- w$scale
    length2 <- w$length
    if (is.null(precision1) || is.na(precision1)) precision1 <- 0
    if (is.null(scale1) || is.na(scale1)) scale1 <- 0
    if (is.null(length1) || is.na(length1)) length1 <- 0
    if (is.null(precision2) || is.na(precision2)) precision2 <- 0
    if (is.null(scale2) || is.na(scale2)) scale2 <- 0
    if (is.null(length2) || is.na(length2)) length2 <- 0
    ctx$emit(precision1, scale1, length1, precision2, scale2, length2)
}
/


create r scalar script
get_char_length(text char(10))
emits(len1 number, len2 number, dummy char(20))
as
run <- function(ctx) {
	v <- exa$meta$input_columns[[1]]
	w <- exa$meta$output_columns[[3]]
	ctx$emit(v$length,w$length,'9876543210')
}
/
