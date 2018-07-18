CREATE lua SCALAR SCRIPT
unicode_len("str" VARCHAR(1000))
RETURNS DOUBLE AS

function run(ctx)
	if ctx.str ~= null then
		return unicode.utf8.len(ctx.str)
	end
end
/

CREATE lua SCALAR SCRIPT
unicode_upper("str" VARCHAR(1000))
RETURNS VARCHAR(1000) AS

function run(ctx)
	if ctx.str ~= null then
		return unicode.utf8.upper(ctx.str)
	end
end
/

CREATE lua SCALAR SCRIPT
unicode_lower("str" VARCHAR(1000))
RETURNS VARCHAR(1000) AS

function run(ctx)
	if ctx.str ~= null then
		return unicode.utf8.lower(ctx.str)
	end
end
/

create lua SCALAR SCRIPT
unicode_count("word" VARCHAR(1000), "convert" double)
EMITS (UCHAR VARCHAR(1), COUNT double) as

function run(ctx)
	word = ctx.word
	if word == null then
		return
	end
	if ctx.convert > 0 then
		word = unicode.utf8.upper(word)
	elseif ctx.convert < 0 then
		word = unicode.utf8.lower(word)
	end
	count = {}
	for i=1, unicode.utf8.len(word) do
		c = unicode.utf8.sub(word, i, i)
		v = count[c]
		if v == nil then
			v = 0
		end
		count[c] = v + 1
	end
	for k, v in pairs(count) do
		ctx.emit(k, v)
	end
end
/



-- vim: ts=2:sts=2:sw=2
