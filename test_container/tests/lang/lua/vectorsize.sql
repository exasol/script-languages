CREATE LUA SCALAR SCRIPT vectorsize5000(A DOUBLE) 
RETURNS VARCHAR(2000000) AS

numbers = {}

for i = 1,5000
do
  table.insert(numbers, tostring(i))
end
ret = table.concat(numbers)

function run(ctx)
  return ret
end
/


CREATE LUA SCALAR SCRIPT
vectorsize(length DOUBLE, dummy DOUBLE) 
RETURNS VARCHAR(2000000) AS

cache = {}

function fill_cache(n) 
  local numbers = {}
  for i = 1, n do
    table.insert(numbers, tostring(i))
  end
  cache[n] = table.concat(numbers)
end

function run(ctx)
  if cache[ctx.length] == nil then
    fill_cache(ctx.length)
  end
  return cache[ctx.length]
end
/

CREATE LUA SCALAR SCRIPT
vectorsize_set(length DOUBLE, n DOUBLE, dummy DOUBLE) 
EMITS (o VARCHAR(2000000)) AS

cache = {}

function fill_cache(n) 
  local numbers = {}
  for i = 1, n do
    table.insert(numbers, tostring(i))
  end
  cache[n] = table.concat(numbers)
end

function run(ctx)
  if cache[ctx.length] == nil then
    fill_cache(ctx.length)
  end
  for i = 1, ctx.n do
    ctx.emit(cache[ctx.length])
  end
end
/

-- vim: ts=2:sts=2:sw=2:et:fdm=indent
