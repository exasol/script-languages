CREATE EXTERNAL SCALAR SCRIPT
cologne_phonetic(word VARCHAR(200))
RETURNS VARCHAR(200) AS
# redirector @@redirector_url@@

import string

def nicht_vor(x):
	def context_handler(word, pos, length):
		if pos == length-1:
			return True
		if pos < length-1:
			if all([word[pos+1] != c for c in x]):
				return True
		return False
	return context_handler

def vor(x):
	def context_handler(word, pos, length):
		if pos < length-1:
			if any([word[pos+1] == c for c in x]):
				return True
		return False
	return context_handler

def vor_ausser_nach(x,y):
	def context_handler(word, pos, length):
		if pos < length-1:
			if any([word[pos+1] == c for c in x]):
				if all([word[pos-1] != c for c in y]):
					return True
		return False
	return context_handler

def nicht_nach(x):
	def context_handler(word, pos, length):
		if pos > 0:
			if all([word[pos-1] != c for c in x]):
				return True
		return False
	return context_handler

def nach(x):
	def context_handler(word, pos, length):
		if pos > 0:
			if any([word[pos-1] == c for c in x]):
				return True
		return False
	return context_handler

def im_anlaut_vor(x):
	def context_handler(word, pos, length):
		if pos == 1:
			if any([word[0] == c for c in x]):
				return True
		return False
	return context_handler

def im_anlaut_ausser_vor(x):
	def context_handler(word, pos, length):
		if pos == 1:
			if all([word[0] != c for c in x]):
				return True
		return False
	return context_handler

RULES = [
	('a',  '0'),
	('e',  '0'),
	('i',  '0'),
	('j',  '0'),
	('o',  '0'),
	('u',  '0'),
	('y',  '0'),
	(u'ö',  '0'),
	(u'ä',  '0'),
	(u'ü',  '0'),
	(u'ß',  '0'),
	('h',  '-'),
	('b',  '1'),
	('p',  '1', [nicht_vor('h')]),
	('d',  '2', [nicht_vor('csz')]),
	('t',  '2', [nicht_vor('csz')]),
	('f',  '3'),
	('v',  '3'),
	('w',  '3'),
	('p',  '3', [vor('h')]),
	('g',  '4'),
	('k',  '4'),
	('q',  '4'),
	('c',  '4', [im_anlaut_vor('ahkloqrux'), vor_ausser_nach('ahkoqux','sz')]),
    ('x', '48', [nicht_nach('ckq')]),
	('l',  '5'),
	('m'   '6'),
	('n'   '6'),
	('r',  '7'),
    ('s'   '8'),
    ('z'   '8'),
    ('c',  '8', [nach('sz'), im_anlaut_ausser_vor('ahkloqrux'), nicht_vor('ahkoqux')]),
  	('d',  '8', [vor('csz')]),
  	('t',  '8', [vor('csz')]),
	('x',  '8', [nach('ckq')])]

def encode(word, pos, length):
	for rule in RULES:
		if word[pos] == rule[0]:
			if len(rule) == 2:
				return rule[1]
			elif any([ctx(word,pos,length) for ctx in rule[2]]):
				return rule[1]
	raise Exception('no match for "%s" in "%s" at pos %d' % (word[pos], word, pos))

valid_characters = string.lowercase + u'äöüß'

def run(ctx):
	if ctx.word is not None:
		word = filter(lambda c: c in valid_characters, ctx.word.lower())
		l = len(word)
		code = []
		for i in range(l):
			c = encode(word, i, l)
			if (i == 0 and c != '-') or (len(code) > 0 and c not in ['0', '-'] and c != code[-1]):
				code.append(c)
		return ''.join(code)
/
