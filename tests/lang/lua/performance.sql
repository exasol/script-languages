create lua scalar script
performance_map_words(w varchar(1000))
emits (w varchar(1000), c double) as

function run(exa)
    local word = exa.w
    if (word ~= null)
    then
        for i in string.gmatch(word, '([%w%p]+)')
        do
            exa.emit(i, 1)
        end
    end
end
/

create lua scalar script
performance_map_unicode_words(w varchar(1000))
emits (w varchar(1000), c double) as

function run(exa)
    local word = exa.w
    if (word ~= null)
    then
        for i in unicode.utf8.gmatch(word, '([%w%p]+)')
        do
            exa.emit(i, 1)
        end
    end
end
/

create lua set script
performance_reduce_counts(w varchar(100), c double)
emits (w varchar(100), c double) as

function run(exa)
    local count = 0
    local word = exa.w
    repeat
        count = count + exa.c
    until not exa.next()
    exa.emit(word, count)
end
/


CREATE LUA SCALAR SCRIPT
performance_map_characters(text VARCHAR(350))
EMITS (w CHAR(1), c DOUBLE) AS
function run(ctx)
    if ctx.text ~= null then
        for i = 1,unicode.utf8.len(ctx.text) do
            ctx.emit(unicode.utf8.sub(ctx.text, i, i), 1)
        end
    end
end
/


CREATE LUA SET SCRIPT
performance_reduce_characters(w CHAR(1), c DOUBLE)
EMITS (w CHAR(1), c DOUBLE) AS

function run(ctx)
    local c = 0
    local w = ctx.w
    if (w  ~= null) then
        repeat
            c = c + 1
        until not ctx.next()
        ctx.emit(w, c)
    end
end
/


-- vim: ts=2:sts=2:sw=2
