create python3 SCALAR SCRIPT
unicode_len(word VARCHAR(1000))
RETURNS INT AS

def run(ctx):
	if ctx.word is not None:
		return len(ctx.word)
/

create python3 SCALAR SCRIPT
unicode_upper(word VARCHAR(1000))
RETURNS VARCHAR(1000) AS

def run(ctx):
        if ctx.word is not None:
                x = ctx.word.upper()
                ex = x.encode('utf8')
                p = ex.find(b'\xcc')
                if p != -1:
                        x = (ex[:p]).decode('utf8')
                return x
/

create python3 SCALAR SCRIPT
unicode_lower(word VARCHAR(1000))
RETURNS VARCHAR(1000) AS

def run(ctx):
        if ctx.word is not None:
                x = ctx.word.lower()
                ex = x.encode('utf8')
                p = ex.find(b'\xcc')
                if p != -1:
                        x = (ex[:p]).decode('utf8')
                return x
/


create python3 SCALAR SCRIPT
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
