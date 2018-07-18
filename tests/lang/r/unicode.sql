CREATE r SCALAR SCRIPT
unicode_len(word VARCHAR(1000))
RETURNS INT AS

run <- function(ctx) {
    word <- ctx$word
    if (!is.na(word)) {
        nchar(word)
    } else NA
}   
/

CREATE r SCALAR SCRIPT
unicode_upper(word VARCHAR(1000))
RETURNS VARCHAR(1000) AS

run <- function(ctx) {
    word <- ctx$word
    if (!is.na(word)) {
        toupper(word)
    } else NA
}
/

CREATE r SCALAR SCRIPT
unicode_lower(word VARCHAR(1000))
RETURNS VARCHAR(1000) AS

run <- function(ctx) {
    word <- ctx$word
    if (!is.na(word)) {
        tolower(word)
    } else NA
}
/

CREATE r SCALAR SCRIPT
unicode_count(word VARCHAR(1000), convert_ INT)
EMITS (uchar VARCHAR(1), count INT) AS

run <- function(ctx) {
    word <- ctx$word
    convert_ <- ctx$convert_
    if (convert_ > 0) {
        word <- toupper(word)
    } else if (convert_ < 0) {
        word <- tolower(word)
    }

    for (c in unlist(strsplit(word, ''))) {
        ctx$emit(c, 1)
    }
}
/

-- vim: ts=4:sts=4:sw=4:et:fdm=indent
