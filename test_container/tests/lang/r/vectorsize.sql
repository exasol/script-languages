CREATE r SCALAR SCRIPT vectorsize5000(A DOUBLE) 
RETURNS VARCHAR(2000000) AS

rv <- paste(as.character(1:5000), collapse='')

run <- function(ctx) {
  rv
}
/


CREATE r SCALAR SCRIPT
vectorsize(length INT, dummy DOUBLE) 
RETURNS VARCHAR(2000000) AS

cache <- list()

get_cache <- function(n) {
  if (is.null(cache[[as.character(n)]])) {
    cache[as.character(n)] <<- paste(as.character(1:n), collapse='')
  }
  cache[[as.character(n)]]
}

run <- function(ctx) {
  get_cache(ctx$length)
}
/

CREATE r SCALAR SCRIPT
vectorsize_set(length INT, n INT, dummy DOUBLE) 
EMITS (o VARCHAR(2000000)) AS

cache <- list()

get_cache <- function(n) {
  if (is.null(cache[[as.character(n)]])) {
    cache[as.character(n)] <<- paste(as.character(1:n), collapse='')
  }
  cache[[as.character(n)]]
}

run <- function(ctx) {
  for (i in 1:ctx$n) {
    ctx$emit(get_cache(ctx$length))
  }
}
/

-- vim: ts=2:sts=2:sw=2:et:fdm=indent
