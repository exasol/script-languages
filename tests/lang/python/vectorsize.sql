CREATE python SCALAR SCRIPT
vectorsize5000(A DOUBLE) 
RETURNS VARCHAR(2000000) AS

retval = ''.join([str(i) for i in range(5000)])

def run(ctx):
    return retval
/


CREATE python SCALAR SCRIPT
vectorsize(length INT, dummy DOUBLE) 
RETURNS VARCHAR(2000000) AS
import gc

cache = {}
cache_size = 0
cache_max = 1024*1024*64

def run(ctx):
    global cache_size, cache
    if ctx.length not in cache:
        curstr = ''.join([str(i) for i in xrange(ctx.length)])
        if cache_size + len(curstr) > cache_max:
            cache = {}
            cache_size = 0
            gc.collect()
        cache[ctx.length] = curstr
        cache_size += len(curstr)
    return cache[ctx.length]
/


CREATE python SCALAR SCRIPT
vectorsize_set(length INT, n INT, dummy DOUBLE) 
EMITS (o VARCHAR(2000000)) AS
import gc

cache = {}
cache_size = 0
cache_max = 1024*1024*64

def run(ctx):
    global cache_size, cache
    if ctx.length not in cache:
        curstr = ''.join([str(i) for i in xrange(ctx.length)])
        if cache_size + len(curstr) > cache_max:
            cache = {}
            cache_size = 0
            gc.collect()
        cache[ctx.length] = curstr
        cache_size += len(curstr)
    for i in xrange(ctx.n):
        ctx.emit(cache[ctx.length])
/



-- vim: ts=4:sts=4:sw=4:et:fdm=indent
