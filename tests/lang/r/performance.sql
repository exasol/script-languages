CREATE R SCALAR SCRIPT
performance_map_words(w VARCHAR(1000))
EMITS (w VARCHAR(1000), c INTEGER) AS
findall <- function(pattern, string) {
    m <- gregexpr(pattern, string, perl=TRUE);
    res <- vector(length=length(m[[1]]));
    for (c in 1:length(m[[1]])) {
        res[c] <- substr(string, attr(m[[1]], "capture.start")[c],  attr(m[[1]], "capture.start")[c]+ attr(m[[1]], "capture.length")[c]);
    };
    res
}

run <- function(ctx) {
    if (!is.na(ctx$w)) {
        for (w in findall('([]\\w!"#$%&\'()*+,./:;<=>?@[\\^_`{|}~-]+)', ctx$w)) {
            ctx$emit(w,1);
        }
    }
}
/

CREATE R SCALAR SCRIPT
performance_map_unicode_words(w VARCHAR(1000))
EMITS (w VARCHAR(1000), c INTEGER) AS


run <- function(ctx) {
    if (!is.na(ctx$w)) {
        for (w in grep('([]\\w!"#$%&\'()*+,./:;<=>?@[\\^_`{|}~-]+)', strsplit(ctx$w,"")[[1]], perl=TRUE, value=TRUE)) {
            ctx$emit(w,1);
        }
    }
}
/

CREATE R SET SCRIPT
performance_reduce_counts(w VARCHAR(1000), c INTEGER)
EMITS (w VARCHAR(1000), c INTEGER) AS
run <- function(ctx) {
    word <- ctx$w;
    count <- 0;
    repeat {
       count <- count + ctx$c;
       if (!ctx$next_row()) break;
    }
    ctx$emit(word, count);
}
/

CREATE R SET SCRIPT
performance_reduce_counts_fast0(w VARCHAR(1000), c DOUBLE)
EMITS (w VARCHAR(1000), c DOUBLE) AS
run <- function(ctx) {
    word <- ctx$w;
    ctx$next_row(NA)
    ctx$emit(word, sum(ctx$c))
}
/

CREATE R SET SCRIPT
performance_reduce_counts_fast7(w VARCHAR(1000), c DOUBLE)
EMITS (w VARCHAR(1000), c DOUBLE) AS
run <- function(ctx) {
  word <- ctx$w;
  count <- 0
  while(ctx$next_row(7))
    count <- count + sum(ctx$c)
  ctx$emit(word, count)
}
/

CREATE R SET SCRIPT
performance_reduce_counts_fast77(w VARCHAR(1000), c DOUBLE)
EMITS (w VARCHAR(1000), c DOUBLE) AS
run <- function(ctx) {
  word <- ctx$w
  count <- 0
  while(ctx$next_row(77))
    count <- count + sum(ctx$c)
  ctx$emit(word, count)
}
/

CREATE R SET SCRIPT
performance_reduce_counts_fast777(w VARCHAR(1000), c DOUBLE)
EMITS (w VARCHAR(1000), c DOUBLE) AS
run <- function(ctx) {
  word <- ctx$w;
  count <- 0
  while(ctx$next_row(77))
    count <- count + sum(ctx$c)
  ctx$emit(word, count)
}
/

CREATE R SET SCRIPT
performance_reduce_counts_fast7777(w VARCHAR(1000), c DOUBLE)
EMITS (w VARCHAR(1000), c DOUBLE) AS
run <- function(ctx) {
  word <- ctx$w;
  count <- 0
  while(ctx$next_row(7777))
    count <- count + sum(ctx$c)
  ctx$emit(word, count)
}
/

CREATE R SET SCRIPT
performance_reduce_counts_fast777777(w VARCHAR(1000), c DOUBLE)
EMITS (w VARCHAR(1000), c DOUBLE) AS
run <- function(ctx) {
  word <- ctx$w;
  count <- 0
  while(ctx$next_row(777777))
    count <- count + sum(ctx$c)
  ctx$emit(word, count)
}
/

CREATE R SET SCRIPT
performance_reduce_counts_fast77777777(w VARCHAR(1000), c DOUBLE)
EMITS (w VARCHAR(1000), c DOUBLE) AS
run <- function(ctx) {
  word <- ctx$w;
  count <- 0
  while(ctx$next_row(77777777))
    count <- count + sum(ctx$c)
  ctx$emit(word, count)
}
/

CREATE R SCALAR SCRIPT
performance_map_characters(text VARCHAR(1000))
EMITS (w CHAR(1), c INTEGER) AS
run <- function(ctx) {
    if (!is.na(ctx$text)) {
        for (c in unlist(strsplit(ctx$text, ''))) {
            ctx$emit(c, 1);
        }
    }
}
/

CREATE R SET SCRIPT
performance_reduce_characters(w CHAR(1), c INTEGER)
EMITS (w CHAR(1), c INTEGER) AS
run <- function(ctx) {
    c <- 0;
    w <- ctx$w;
    if (!is.na(w)) {
        repeat {
            c <- c+1;
            if (!ctx$next_row()) break;
        }
        ctx$emit(w,c);
    }
}
/

CREATE R SCALAR SCRIPT
performance_map_characters_fast(text VARCHAR(1000))
EMITS (w CHAR(1), c DOUBLE) AS
run <- function(ctx) {
  ctx$emit(strsplit(ctx$text, '')[[1]], 1)
}
/

CREATE R SCALAR SCRIPT
performance_map_characters_fast0(text VARCHAR(1000))
EMITS (w CHAR(1), c DOUBLE) AS
run <- function(ctx) {
  ctx$next_row(NA)
  ctx$emit(unlist(strsplit(ctx$text, '')), 1)
}
/

CREATE R SET SCRIPT
performance_reduce_characters_fast(w CHAR(1), c DOUBLE)
EMITS (w CHAR(1), c DOUBLE) AS
run <- function(ctx) {
  word <- ctx$w;
  ctx$next_row(NA)
  ctx$emit(word, sum(ctx$c))
}
/

-- vim: ts=2:sts=2:sw=2
