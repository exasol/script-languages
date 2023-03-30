CREATE r SCALAR SCRIPT
basic_emit_several_groups(a INTEGER, b INTEGER)
EMITS (i INTEGER, j VARCHAR(40)) AS
run <- function(ctx) {
    for(n in seq(ctx$a)) {
        for(i in seq(ctx$b)) {
            ctx$emit(i-1, exa$meta$vm_id)
        }
    }
}
/

CREATE r SET SCRIPT
basic_test_reset(i INTEGER, j VARCHAR(40))
EMITS (k INTEGER) AS
run <- function(ctx) {
    ctx$emit(ctx$i)
    ctx$next_row()
    ctx$emit(ctx$i)
    ctx$reset()
    ctx$emit(ctx$i)
    ctx$next_row()
    ctx$emit(ctx$i)
}
/

CREATE r SCALAR SCRIPT
basic_emit_two_ints()
EMITS (i INTEGER, j INTEGER) AS

run <- function(ctx) {
    ctx$emit(1, 2)
}
/


CREATE r SCALAR SCRIPT
basic_nth_partial_sum(n INTEGER)
RETURNS INTEGER as

run <- function(ctx) {
    if (!is.na(ctx$n)) {
        ctx$n * (ctx$n + 1) / 2
    } else {
        0
    }
}
/

    
CREATE r SCALAR SCRIPT
basic_range(n INTEGER)
EMITS (n INTEGER) AS

run <- function(ctx) {
    if (!is.na(ctx$n)) {
        for (i in 0:(ctx$n-1)) {
            ctx$emit(i)
        }
    }
}
/


CREATE r SET SCRIPT
basic_sum(x INTEGER)
RETURNS INTEGER AS

run <- function(ctx) {
    s = 0
    repeat {
        if (!is.na(ctx$x)) {
            s = s + ctx$x
        }
        if (!ctx$next_row()) break
    }
    s
}
/


CREATE r SET SCRIPT
basic_sum_grp(x INTEGER)
EMITS (s INTEGER) AS

run <- function(ctx) {
    s = 0
    repeat {
        if (!is.na(ctx$x)) {
            s = s + ctx$x
        }
        if (!ctx$next_row()) break
    }
    ctx$emit(s)
}
/

CREATE r SET SCRIPT
set_returns_has_empty_input(a double) RETURNS boolean AS
run <- function (ctx) {
    if (is.na(ctx$x)) {
        TRUE
    } else {
        FALSE
    }
}
/

CREATE r SET SCRIPT
set_emits_has_empty_input(a double) EMITS (x double, y varchar(10)) AS
run <- function (ctx) {
    if (is.na(ctx$x)) {
        ctx$emit(1,'1')
    } else {
        ctx$emit(2,'2')
    }
}
/

-- vim: ts=4:sts=4:sw=4:et:fdm=indent
