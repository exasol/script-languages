CREATE OR REPLACE R SCALAR SCRIPT
DEFAULT_TZ()
RETURNS VARCHAR(100) AS

run <- function(ctx) {
    non_dst_date <- as.POSIXct("2023-01-01 12:00:00")
    timezone_short_name <- format(non_dst_date, "%Z")
    timezone_short_name
}
/

CREATE OR REPLACE R SCALAR SCRIPT
MODIFY_TZ_TO_NEW_YORK()
RETURNS VARCHAR(100) AS

run <- function(ctx) {
    Sys.setenv("TZ" = "America/New_York")
    non_dst_date <- as.POSIXct("2023-01-01 12:00:00")
    timezone_short_name <- format(non_dst_date, "%Z")
    timezone_short_name
}
/
