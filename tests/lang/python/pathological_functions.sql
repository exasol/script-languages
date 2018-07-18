CREATE python SCALAR SCRIPT
sleep("sec" double)
RETURNS double AS

import time

def run(ctx):
    time.sleep(ctx.sec)
    return ctx.sec
/

CREATE python SCALAR SCRIPT
mem_hog("mb" int)
RETURNS int AS

def run(ctx):
    a = 'x' * 1024 * 1024
    b = {}
    for i in range(ctx.mb):
            b[i] = '%s%d' % (a+'y', i)
    return len(b)
/

CREATE python SCALAR SCRIPT
cleanup_check("raise_exc" boolean, "sleep" int)
RETURNS int AS

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
