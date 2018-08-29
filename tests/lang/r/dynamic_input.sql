CREATE R SCALAR SCRIPT
metadata_scalar_emit (...)
EMITS("v" VARCHAR(2000)) AS
run <- function(ctx) {
    ctx$emit(paste(exa$meta$input_column_count))
    for(i in seq(exa$meta$input_column_count)) {
        ctx$emit(paste(exa$meta$input_columns[[i]]$name))
        ctx$emit(paste(exa$meta$input_columns[[i]]$type))
        ctx$emit(paste(exa$meta$input_columns[[i]]$sql_type))
        ctx$emit(paste(exa$meta$input_columns[[i]]$precision))
        ctx$emit(paste(exa$meta$input_columns[[i]]$scale))
        ctx$emit(paste(exa$meta$input_columns[[i]]$length))
    }
}
/

CREATE R SCALAR SCRIPT
metadata_scalar_return (...)
RETURNS VARCHAR(2000) AS
run <- function(ctx) {
    paste(exa$meta$input_column_count)
}
/

CREATE R SCALAR SCRIPT
basic_scalar_emit( ... )
EMITS ("v" VARCHAR(2000)) as
run <- function(ctx) {
    for(i in seq(exa$meta$input_column_count))
        ctx$emit(paste(ctx[[i]](), collapse = "."))
}
/

CREATE R SCALAR SCRIPT
basic_scalar_return( ... )
RETURNS VARCHAR(2000) as
run <- function(ctx) {
    paste(ctx[[exa$meta$input_column_count]]())
}
/

CREATE R SET SCRIPT
basic_set_emit( ... )
EMITS ("v" VARCHAR(2000)) as
run <- function(ctx) {
    var = "result: "
    repeat {
        for(i in seq(exa$meta$input_column_count)) {
            ctx$emit(paste(ctx[[i]](), collapse = "."))
            var = paste(var, paste(ctx[[i]](), collapse = "."), sep = " , ")
        }
        if(!ctx$next_row()) break
    }
    ctx$emit(var)
}
/

CREATE R SET SCRIPT
basic_set_return( ... )
RETURNS VARCHAR(2000) as
run <- function(ctx) {
    var = 'result: '
    repeat {
        for(i in seq(exa$meta$input_column_count))
            var = paste(var, paste(ctx[[i]](), collpase = ""), sep = " , ")
        if(!ctx$next_row()) break
    }
    var
}
/

CREATE R SET SCRIPT
type_specific_add(...)
RETURNS VARCHAR(2000) as
run <- function(ctx) {
    var = 'result: '
    if(exa$meta$input_columns[[1]]$type == 'character') {
        repeat {
            for(i in seq(exa$meta$input_column_count))
                var = paste(var, ctx[[i]](), sep = " , ")
            if(!ctx$next_row()) break
        }
    } else {
        sum = 0
        repeat {
            for(i in seq(exa$meta$input_column_count))
                sum = sum + ctx[[i]]()
            if(!ctx$next_row()) break
        }
        var = paste(var, sum)
    }
    var
}
/

CREATE R SCALAR SCRIPT
wrong_arg(...)
returns varchar(2000) as
run <- function(ctx) {
    paste(ctx[[2]](), collapse = ",")
}
/

CREATE R SCALAR SCRIPT
wrong_operation(...)
returns varchar(2000) as
run <- function(ctx) {
    paste(ctx[[1]]() * ctx[[2]](), collapse = ",")
}
/

CREATE R SET SCRIPT
empty_set_returns( ... )
returns varchar(2000) as
run <- function(ctx) {
    "1"
}
/

CREATE R SET SCRIPT
empty_set_emits( ... )
emits (x varchar(2000)) as
run <- function(ctx) {
    "1"
}
/
