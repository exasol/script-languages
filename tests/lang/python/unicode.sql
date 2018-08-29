CREATE python SCALAR SCRIPT
unicode_len(word VARCHAR(1000))
RETURNS INT AS

def run(ctx):
	if ctx.word is not None:
		return len(ctx.word)
/

CREATE python SCALAR SCRIPT
unicode_upper(word VARCHAR(1000))
RETURNS VARCHAR(1000) AS

def run(ctx):
	if ctx.word is not None:
		return ctx.word.upper()
/

CREATE python SCALAR SCRIPT
unicode_lower(word VARCHAR(1000))
RETURNS VARCHAR(1000) AS

def run(ctx):
	if ctx.word is not None:
		return ctx.word.lower()
/


CREATE python SCALAR SCRIPT
unicode_count(word VARCHAR(1000), convert_ INT)
EMITS (uchar VARCHAR(1), count INT) AS

import collections

def run(ctx):
	if ctx.convert_ > 0:
		word = ctx.word.upper()
	elif ctx.convert_ < 0:
		word = ctx.word.lower()
	else:
		word = ctx.word

	count = collections.Counter(word)
	for k, v in count.items():
		ctx.emit(k, v)
/

-- vim: ts=2:sts=2:sw=2
