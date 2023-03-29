
CREATE R SET SCRIPT VAREMIT_SIMPLE_SET (a double) EMITS (...) AS
run <- function(ctx) {
  ctx$emit(1)
}
/

CREATE R SCALAR SCRIPT VAREMIT_SIMPLE_SCALAR (a double) EMITS (...) AS
run <- function(ctx) {
  ctx$emit(1)
}
/

CREATE R SCALAR SCRIPT VAREMIT_SIMPLE_ALL_DYN (...) EMITS (...) AS
run <- function(ctx) {
  ctx$emit(1)
}
/

CREATE R SET SCRIPT VAREMIT_SIMPLE_SYNTAX_VAR (...) EMITS ( ...   ) AS
run <- function(ctx) {
  ctx$emit(1)
}
/

CREATE R SET SCRIPT VAREMIT_GENERIC_EMIT (a varchar(100)) EMITS (...) AS
run <- function(ctx) {
  outRec <- list()
  for (i in 1:exa$meta$output_column_count) {
    outRec[i] <- ctx[[1]]()
  }
  do.call(ctx$emit, outRec)
}
/

CREATE R SET SCRIPT VAREMIT_ALL_GENERIC (...) EMITS (...) AS
run <- function(ctx) {
  outRec <- list()
  for (i in 1:exa$meta$output_column_count) {
    outRec[i] <- ctx[[1]]()
  }
  do.call(ctx$emit, outRec)
}
/

CREATE R SET SCRIPT VAREMIT_METADATA_SET_EMIT (...) EMITS(...) AS
run <- function(ctx) {
    ctx$emit(paste(exa$meta$output_column_count), 1)
    for(i in seq(exa$meta$output_column_count)) {
        ctx$emit(paste(exa$meta$output_columns[[i]]$name), 1)
        ctx$emit(paste(exa$meta$output_columns[[i]]$type), 1)
        ctx$emit(paste(exa$meta$output_columns[[i]]$sql_type), 1)
        ctx$emit(paste(exa$meta$output_columns[[i]]$precision), 1)
        ctx$emit(paste(exa$meta$output_columns[[i]]$scale), 1)
        ctx$emit(paste(exa$meta$output_columns[[i]]$length), 1)
    }
}
/

CREATE R SET SCRIPT VAREMIT_NON_VAR_EMIT (...) EMITS (a double) AS
run <- function(ctx) {
  ctx$emit(1)
}
/

CREATE R SET SCRIPT VAREMIT_SIMPLE_RETURNS (a int) RETURNS int AS
run <- function(ctx) {
  ctx$emit(1)
}
/

CREATE R SET SCRIPT VAREMIT_EMIT_INPUT (...) EMITS (...) AS
run <- function(ctx) {
  outRec <- list()
  for (i in 1:exa$meta$output_column_count) {
    outRec[i] <- ctx[[i]]()
  }
  do.call(ctx$emit, outRec)
}
/

CREATE R SET SCRIPT VAREMIT_EMIT_INPUT_WITH_META_CHECK (...) EMITS (...) AS
run <- function(ctx) {
  outRec <- list()
  for (i in 1:exa$meta$output_column_count) {
    outRec[i] <- ctx[[i]]()
  }
  do.call(ctx$emit, outRec)
}
/

CREATE R SET SCRIPT DEFAULT_VAREMIT_EMPTY_DEF(X DOUBLE) EMITS (...) AS
run <- function(ctx) {
    ctx$emit(1.4)
}

defaultOutputColumns <- function() {
    return ("")
}
/

