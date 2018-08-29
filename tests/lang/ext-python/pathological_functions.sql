CREATE EXTERNAL SCALAR SCRIPT
sleep("sec" double)
RETURNS double AS
# redirector @@redirector_url@@

import time

def run(ctx):
    time.sleep(ctx.sec)
    return ctx.sec
/

CREATE EXTERNAL SCALAR SCRIPT
mem_hog("mb" int)
RETURNS int AS
# redirector @@redirector_url@@

def run(ctx):
    a = 'x' * 1024 * 1024
    b = {}
    for i in range(ctx.mb):
        b[i] = a * i
    return len(b)
/

CREATE EXTERNAL SCALAR SCRIPT
cleanup_check("raise_exc" boolean, "sleep" int)
RETURNS int AS
# redirector @@redirector_url@@

import time
sleep = 0

def run(ctx):
    global sleep
    sleep = ctx.sleep
    if ctx.raise_exc:
        raise ValueError()
    return 42

def cleanup():
    time.sleep(sleep)
    
/


-- vim: ts=4:sts=4:sw=4:et:fdm=indent
