
create r set script
SET_RETURNS(x double, y double)
returns double as

run <- function(ctx) {
  acc <- 0.0
  repeat {
    acc <- acc + ctx$x + ctx$y
    if (!ctx$next_row()) break
  }
  acc
}
/


create r set script
SET_EMITS(x double, y double)
emits (x double, y double) as

run <- function(ctx) {
  repeat {
    ctx$emit(ctx$y, ctx$x)
    if (!ctx$next_row()) break
  }
}
/


create r scalar script
SCALAR_RETURNS(x double, y double)
returns double as

run <- function(ctx) {
    ctx$x + ctx$y
}
/


create r scalar script
SCALAR_EMITS(x double, y double)
emits (x double, y double) as

run <- function(ctx) {
  x <- ctx$x
  y <- ctx$y
  if (x <= y) {
    for (i in x:y) {
      ctx$emit(i, i * i)
    }
  }
}
/

-- vim: ts=2:sts=2:sw=2:et
