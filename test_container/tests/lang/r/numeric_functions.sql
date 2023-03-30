CREATE r SCALAR SCRIPT
pi()
RETURNS double AS

run <- function(ctx) {
  pi
}
/

CREATE r SCALAR SCRIPT
double_mult("x" double, "y" double)
RETURNS double AS

run <- function(ctx) {
  if (is.na(ctx$x) || is.na(ctx$y)) {
    NA
  } else {
    ctx$x * ctx$y
  }
}
/

CREATE r SCALAR SCRIPT
add_two_doubles(x DOUBLE, y DOUBLE) RETURNS DOUBLE AS

run <- function(ctx) {
  if (is.na(ctx$x) || is.na(ctx$y)) {
    NA
  } else {
    ctx$x + ctx$y
  }
}
/

CREATE r SCALAR SCRIPT
add_three_doubles(x DOUBLE, y DOUBLE, z DOUBLE)
RETURNS DOUBLE AS

run <- function(ctx) {
  if (is.na(ctx$x) || is.na(ctx$y) || is.na(ctx$z)) {
    NA
  } else {
    ctx$x + ctx$y + ctx$z
  }
}
/

CREATE r SCALAR SCRIPT
split_integer_into_digits("x" INTEGER)
EMITS (y INTEGER) AS

run <- function(ctx) {
  if (! is.na(ctx$x)) {
    y <- abs(ctx$x)
    while (y > 0) {
      ctx$emit(y %% 10)
      y <- trunc(y / 10)
    }
  }
}
/

-- vim: ts=2:sts=2:sw=2:et
