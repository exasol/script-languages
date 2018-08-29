CREATE r SCALAR SCRIPT echo_boolean(x BOOLEAN) RETURNS BOOLEAN AS
run <- function(ctx) {
    ctx$x
}
/

CREATE r SCALAR SCRIPT echo_char1(x CHAR(1)) RETURNS CHAR(1) AS
run <- function(ctx) {
    x <- ctx$x
    if (!is.na(x)) { if (nchar(x) == 1) x else NA } else { NA }
}
/



CREATE r SCALAR SCRIPT echo_char10(x CHAR(10)) RETURNS CHAR(10) AS
run <- function(ctx) {
    x <- ctx$x
    if (!is.na(x)) { if (nchar(x) == 10) x else NA } else { NA }
}
/



CREATE r SCALAR SCRIPT echo_date(x DATE) RETURNS DATE AS
run <- function(ctx) {
    ctx$x
}
/


CREATE r SCALAR SCRIPT echo_integer(x INTEGER) RETURNS INTEGER AS
run <- function(ctx) {
    ctx$x
}
/


CREATE r SCALAR SCRIPT echo_double(x DOUBLE) RETURNS DOUBLE AS
run <- function(ctx) {
    ctx$x
}
/

CREATE r SCALAR SCRIPT echo_decimal_36_0(x DECIMAL(36,0)) RETURNS DECIMAL(36,0) AS
run <- function(ctx) {
    ctx$x
}
/

CREATE r SCALAR SCRIPT echo_decimal_36_36(x DECIMAL(36,36)) RETURNS DECIMAL(36,36) AS
run <- function(ctx) {
    ctx$x
}
/

CREATE r SCALAR SCRIPT echo_varchar10(x VARCHAR(10)) RETURNS VARCHAR(10) AS
run <- function(ctx) {
    ctx$x
}
/

CREATE r SCALAR SCRIPT echo_timestamp(x TIMESTAMP) RETURNS TIMESTAMP AS
run <- function(ctx) {
    ctx$x
}
/


-- it is not supportet anymore in R
-- CREATE r SCALAR SCRIPT run_func_is_empty() RETURNS DOUBLE AS
-- run <- function(ctx) {}
-- /


CREATE r SCALAR SCRIPT
bottleneck_varchar10(i VARCHAR(20))
RETURNS VARCHAR(10) AS

run <- function(ctx) {
    ctx$i
}
/

CREATE r SCALAR SCRIPT
bottleneck_char10(i VARCHAR(20))
RETURNS CHAR(10) AS

run <- function(ctx) {
    ctx$i
}
/

CREATE r SCALAR SCRIPT
bottleneck_decimal5(i DECIMAL(20, 0))
RETURNS DECIMAL(5, 0) AS

run <- function(ctx) {
    ctx$i
}
/

-- vim: ts=4:sts=4:sw=4:et:fdm=indent
