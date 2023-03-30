CREATE R SCALAR SCRIPT
sleep(sec double)
RETURNS double AS
run <- function(ctx) {
    p1 <- proc.time();
    Sys.sleep(ctx$sec);
    p2 <- proc.time() - p1;
    p2[3];
}
/


CREATE R SCALAR SCRIPT
mem_hog(mb int)
RETURNS int AS
run <- function(ctx) {
    a <- paste(rep("x",1024*1024),collapse="")
    b <- vector(length=ctx$mb);
    for (i in 1:ctx$mb) {
        b[[i]] <- paste(a,"y",i,sep="")
    }
    length(b)
}
/

CREATE R SCALAR SCRIPT
cleanup_check(raise_exc boolean, sleep int)
RETURNS int AS

sleep <- 0

run <- function(ctx) {
    sleep <<- ctx$sleep;
    if (ctx$raise_exc) {
        stop('Value Error');
    }
    42
}

cleanup <- function() {
    Sys.sleep(5)
}
/


-- vim: ts=2:sts=2:sw=2
