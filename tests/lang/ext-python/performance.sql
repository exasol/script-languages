
CREATE EXTERNAL SCALAR SCRIPT
performance_map_words(w VARCHAR(1000))
EMITS (w VARCHAR(1000), c INTEGER) AS
# redirector @@redirector_url@@

import re
import string

pattern = re.compile(r'''([]\w!"#$%&\'()*+,./:;<=>?@[\\^_`{|}~-]+)''')

def run(ctx):
	if ctx.w is not None:
		for w in re.findall(pattern, ctx.w):
			ctx.emit(w, 1)
/

CREATE EXTERNAL SCALAR SCRIPT
performance_map_unicode_words(w VARCHAR(1000))
EMITS (w VARCHAR(1000), c INTEGER) AS
# redirector @@redirector_url@@

import re
import string

pattern = re.compile(r'''([]\w!"#$%&\'()*+,./:;<=>?@[\\^_`{|}~-]+)''', re.UNICODE)

def run(ctx):
	if ctx.w is not None:
		for w in re.findall(pattern, ctx.w):
			ctx.emit(w, 1)
/

CREATE EXTERNAL SET SCRIPT
performance_reduce_counts(w VARCHAR(1000), c INTEGER)
EMITS (w VARCHAR(1000), c INTEGER) AS
# redirector @@redirector_url@@

def run(ctx):
	word = ctx.w
	count = 0
	while True:
		count += ctx.c
		if not ctx.next(): break
	ctx.emit(word, count)
/


CREATE EXTERNAL SCALAR SCRIPT
performance_map_characters(text VARCHAR(1000))
EMITS (w CHAR(1), c INTEGER) AS
# redirector @@redirector_url@@
def run(ctx):
    if ctx.text is not None:
        for c in ctx.text:
            ctx.emit(c, 1)
/


CREATE EXTERNAL SET SCRIPT
performance_reduce_characters(w CHAR(1), c INTEGER)
EMITS (w CHAR(1), c INTEGER) AS
# redirector @@redirector_url@@

def run(ctx):
    c = 0
    w = ctx.w
    if w is not None:
        while True:
            c += 1
            if not ctx.next(): break
        ctx.emit(w, c)
/

-- vim: ts=2:sts=2:sw=2
