#!/usr/bin/env python2
# coding: utf-8

import csv
import locale
import logging
import os
import subprocess
import sys
import tempfile
import unicodedata
import re
import argparse

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))


import udf
udf.pythonVersionInUdf = -1
from udf import (
    requires,
    useData,
    expectedFailureIfLang,
    )

from exatest.testcase import skipIf

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def utf8encoder(row):
    return tuple(
            (x.encode('utf-8') if isinstance(x, unicode) else x)
             for x in row)


def getPythonVersionInUDFs(server,script_languages):
        log = logging.getLogger('unicodedata')
        log.info("trying to figure out python version of python in UDFs")
        sql = udf.fixindent('''
           alter session set script_languages='%(sl)s';
           drop schema if exists pyversion_schema cascade;
           create schema pyversion_schema;
           create or replace python scalar script pyversion_schema.python_version() returns varchar(1000) as
           import sys
           def run(ctx):
               return 'Python='+str(sys.version_info[0])
           /
           select pyversion_schema.python_version();
           ''' % {'sl':script_languages})
        cmd = '''%(exaplus)s -c %(conn)s -u sys -P exasol
		        -no-config -autocommit ON -L -pipe''' % {
			        'exaplus': os.environ.get('EXAPLUS',
				        '/usr/opt/EXASuite-4/EXASolution-4.2.9/bin/Console/exaplus'),
			        'conn': server
			        }
        env = os.environ.copy()
        #env['PATH'] = '/usr/opt/jdk1.8.0_latest/bin:' + env['PATH']
        exaplus = subprocess.Popen(
                cmd.split(),
                env=env,

                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
        out, _err = exaplus.communicate(sql)
        pythonVersionInUdf = -1
        for line in out.strip().split('\n'):
            m = re.search(r'Python=(\d)',line)
            if m:
                pythonVersionInUdf = int(m.group(1))
                continue

                
        if pythonVersionInUdf not in [2,3]:
            print('cannot set pythonVersionInUdf: %s' % pythonVersionInUdf)
            sys.exit(1)

        return pythonVersionInUdf

def setUpModule():
    log = logging.getLogger('unicodedata')

    log.info('generating unicodedata CSV')
    with tempfile.NamedTemporaryFile(prefix='unicode-', suffix='.csv') as csvfile:
        c = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        for i in xrange(sys.maxunicode+1):
            if i>= 5024 and i<=5119:
                continue   # the Unicode Cherokee-Block is broken in Python 2.7 and Python 3.4 (maybe also 3.5)
            u = unichr(i)
            if unicodedata.category(u).startswith('C'):
                # [Cc]Other, Control
                # [Cf]Other, Format
                # [Cn]Other, Not Assigned
                # [Co]Other, Private Use
                # [Cs]Other, Surrogate
                continue
            row = (i,                                       # INT 0-1114111
                unicodedata.name(u, 'UNICODE U+%08X' % i),  # VARCHAR(100) ASCII
                u,                                          # VARCHAR(1) UNICODE
                u.upper(),                                  # VARCHAR(1) UNICODE
                u.lower(),                                  # VARCHAR(1) UNICODE
                unicodedata.decimal(u, None),               # INT
                unicodedata.numeric(u, None),               # DOUBLE
                unicodedata.category(u),                    # VARCHAR(3) ASCII
                unicodedata.bidirectional(u),               # VARCHAR(3) ASCII
                unicodedata.combining(u),                   # VARCHAR(3) ASCII
                unicodedata.east_asian_width(u),            # VARCHAR(1) ASCII
                bool(unicodedata.mirrored),                 # BOOLEAN
                unicodedata.decomposition(u),               # VARCHAR(10) ASCII
                unicodedata.normalize('NFC', u),            # VARCHAR(3) UNICODE
                unicodedata.normalize('NFD', u),            # VARCHAR(3) UNICODE
                unicodedata.normalize('NFKC', u),           # VARCHAR(3) UNICODE
                unicodedata.normalize('NFKD', u),           # VARCHAR(3) UNICODE
                )
            c.writerow(utf8encoder(row))
        csvfile.flush()

        log.info('loading CSV')
        sql = '''
            DROP SCHEMA utest CASCADE;
            CREATE SCHEMA utest;
            CREATE TABLE unicodedata (
                codepoint INT NOT NULL,
                name VARCHAR(100) ASCII,
                uchar VARCHAR(1) UTF8,
                to_upper VARCHAR(1) UTF8,
                to_lower VARCHAR(1) UTF8,
                decimal_value INT,
                numeric_value INT,
                category VARCHAR(3) ASCII,
                bidirectional VARCHAR(3) ASCII,
                combining VARCHAR(10) ASCII,
                east_asian_width VARCHAR(2) ASCII,                mirrored BOOLEAN,
                decomposition VARCHAR(100) ASCII,
                NFC VARCHAR(10) UTF8,
                NFD VARCHAR(10) UTF8,
                NFKC VARCHAR(20) UTF8,
                NFKD VARCHAR(20) UTF8
                );
            IMPORT INTO unicodedata
            FROM LOCAL CSV FILE '%s'
            ROW SEPARATOR = 'CRLF';
            ''' % os.path.join(os.getcwd(), csvfile.name)
        cmd = '''%(exaplus)s -c %(conn)s -u sys -P exasol
		        -no-config -autocommit ON -L -pipe''' % {
			        'exaplus': os.environ.get('EXAPLUS',
				        '/usr/opt/EXASuite-4/EXASolution-4.2.9/bin/Console/exaplus'),
			        'conn': udf.opts.server
			        }
        env = os.environ.copy()
        env['PATH'] = '/usr/opt/jdk1.8.0_latest/bin:' + env['PATH']
        exaplus = subprocess.Popen(
                cmd.split(),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
        out, _err = exaplus.communicate(sql)
    if exaplus.returncode != 0:
        log.critical('EXAplus error: %d', exaplus.returncode)
        log.error(out)
    else:
        log.debug(out)


def add_uniname(data):
    return [(n, unicodedata.name(unichr(n), 'U+%04X' % n))
            for n in data]

class Unicode(udf.TestCase):

    def query_unicode_char(self, u):
        rows = self.query('''
            SELECT count, unicode(uchar) AS u
            FROM (
                SELECT fn1.unicode_count(unicodechr(%d), 0)
                FROM dual)
            ''' % u)
        self.assertEqual(1, self.rowcount())
        self.assertEqual(1, rows[0].COUNT)
        self.assertEqual(u, rows[0].U)

    data = add_uniname((
                65,
                255,
                382,
                65279,
                63882,
                65534, 
                66432, 
                173746, 
                1114111,
                ))
   
    @useData(data) 
    @requires('UNICODE_COUNT')
    def test_unicode(self, codepoint, _name):
        self.query_unicode_char(codepoint)

    @requires('UNICODE_LEN')
    def test_unicode_count(self):
        self.maxDiff = 1024
        rows = self.query('''
            SELECT
                c1_integer AS i,
                len(c2_varchar100) AS len_exa,
                fn1.unicode_len(c2_varchar100) AS len
            FROM test.enginetablebigunicodevarchar
            WHERE len(c2_varchar100) != fn1.unicode_len(c2_varchar100)
            ORDER BY c1_integer
            LIMIT 100
            ''')
        self.assertRowsEqual([], rows)

class UnicodeData(udf.TestCase):

    @requires('UNICODE_LOWER')
    #@expectedFailureIfLang('java')
    #@expectedFailureIfLang('r')
    def test_unicode_lower_is_subset_of_Unicode520_part1(self):
        if udf.pythonVersionInUdf == 2:
           rows = self.query('''
               SELECT
                   uchar,
                   to_lower,
                   codepoint,
                   name,
                   unicode(to_lower),
                   unicode(fn1.unicode_lower(uchar))
               FROM utest.unicodedata
               WHERE codepoint not between 55296 and 57343
                   and codepoint not in (8486, 8490, 8491, 304)
                   and (to_lower != fn1.unicode_lower(uchar))
                   and (uchar != fn1.unicode_lower(uchar))
               ORDER BY codepoint
               LIMIT 50
               ''')
           self.maxDiff=None
           self.assertRowsEqual([], rows)

    @requires('UNICODE_LOWER')
    #@expectedFailureIfLang('java')
    #@expectedFailureIfLang('r')
    #@skipIf(udf.pythonVersionInUdf == 3, 'Unicode test does not work in Python3 ... investigate!')
    def test_unicode_lower_is_subset_of_Unicode520_part1_on_undefined_block(self):
        '''DWA-19940 (R)'''
        if udf.pythonVersionInUdf == 2:
            rows = self.query('''
                SELECT
                    codepoint,
                    name,
                    unicode(to_lower),
                    unicode(fn1.unicode_lower(uchar))
                FROM utest.unicodedata
                WHERE codepoint not in (8486, 8490, 8491)
                    and (to_lower != fn1.unicode_lower(uchar))
                    and (uchar != fn1.unicode_lower(uchar))
                    and codepoint between 55296 and 57343
                ORDER BY codepoint
                LIMIT 50
                ''')
            self.assertRowsEqual([], rows)

    @requires('UNICODE_LOWER')
    #@expectedFailureIfLang('lua')
    #@expectedFailureIfLang('java')
    #@expectedFailureIfLang('python3')
    @skipIf(udf.pythonVersionInUdf == 3, 'Unicode test does not work in Python3 ... investigate!')
    def test_unicode_lower_is_subset_of_Unicode520_part2(self):
        '''DWA-13702 (Lua)'''
        if udf.pythonVersionInUdf == 2:
            rows = self.query('''
                SELECT
                    codepoint,
                    name,
                    unicode(to_lower),
                    unicode(fn1.unicode_lower(uchar))
                FROM utest.unicodedata
                WHERE codepoint in (8486, 8490, 8491)
                    and (to_lower != fn1.unicode_lower(uchar))
                    and (uchar != fn1.unicode_lower(uchar))
                ORDER BY codepoint
                LIMIT 50
                ''')
            self.assertRowsEqual([], rows)

    @requires('UNICODE_UPPER')
    #@expectedFailureIfLang('java')
    #@expectedFailureIfLang('r')
    #@skipIf(udf.pythonVersionInUdf == 3, 'Unicode tests do not work in Python3 ... investigate!')
    def test_unicode_upper_is_subset_of_Unicode520_part1(self):
        if udf.pythonVersionInUdf == 2:
            rows = self.query('''
                SELECT
                    codepoint,
                    name,
                    unicode(to_upper)
                    --,
                    --unicode(fn1.unicode_upper(uchar)) 
                FROM utest.unicodedata
                WHERE codepoint not between 55296 and 57343
                    and codepoint not in (181, 1010, 8126, 304)
                    and codepoint not in (223,329,496,604,609,613,614,618,620,647,669,670,912,912,944,1011,1415,7830,7831,7832,7833,7834,8016,8018,8020,8022,8064,8072,8065,8073,8066,8074,8067,8075,8068,8076,8069,8077,8070,8078,8071,8079,8072,8073,8074,8075,8076,8077,8078,8079,8080,8088,8081,8089,8090,8091,8084,8092,8085,8093,8086,8094,8087,8095,8088)
                    and codepoint not in (8082,8090,8091,8096,8104,8097,8105,8098,8106,8099,8107,8100,8108,8101,8109,8102,8110,8103,8111,8104,8105,8106,8107,8108,8109,8110,8111,8114,8115,8116,8118,8119,8124,8130,8131,8132,8134,8135,8140,8146,8147,8150,8151,8162,8163,8164,8166,8167,8178,8179,8188,8180,8182,8183,8188,64256,64257,64258,64259,64260)
                    and codepoint not in (8083,8091,64261,64262,64275,64276,64277,64278,64279)
                    and (to_upper != fn1.unicode_upper(uchar))
                    and (uchar != fn1.unicode_upper(uchar))
                ORDER BY codepoint
                LIMIT 500
                ''')
            self.maxDiff=None
            self.assertRowsEqual([], rows)





    @requires('UNICODE_UPPER')
    #@expectedFailureIfLang('java')
    #@expectedFailureIfLang('r')
#    @skipIf(udf.pythonVersionInUdf == 3, 'Unicode tests do not work in Python3 ... investigate!')
    def test_unicode_upper_is_subset_of_Unicode520_part1_on_undefined_block(self):
        '''DWA-19940 (R)'''
        if udf.pythonVersionInUdf == 2:
           rows = self.query('''
                SELECT
                    codepoint,
                    name,
                    unicode(to_upper),
                    unicode(fn1.unicode_upper(uchar))
                FROM utest.unicodedata
                WHERE codepoint not in (181, 1010, 8126, 304)
                    and (to_upper != fn1.unicode_upper(uchar))
                    and (uchar != fn1.unicode_upper(uchar))
                    and codepoint between 55296 and 57343
                ORDER BY codepoint
                LIMIT 50
                ''')
           self.assertRowsEqual([], rows)

    @requires('UNICODE_UPPER')
    #@expectedFailureIfLang('lua')
    def test_unicode_upper_is_subset_of_Unicode520_part2(self):
        '''DWA-13388 (Lua); DWA-13702 (Lua)'''
        rows = self.query('''
            SELECT
                codepoint,
                name,
                unicode(to_upper),
                unicode(fn1.unicode_upper(uchar))
            FROM utest.unicodedata
            WHERE codepoint in (181, 8126)
                and (to_upper != fn1.unicode_upper(uchar))
                and (uchar != fn1.unicode_upper(uchar))
            ORDER BY codepoint
            LIMIT 50
            ''')
        self.assertRowsEqual([], rows)

    @requires('UNICODE_UPPER')
    @expectedFailureIfLang('lua')
    def test_unicode_upper_is_subset_of_Unicode520_part3(self):
        '''DWA-13388 (Lua); DWA-13702 (Lua); DWA-13782 (R)'''
        rows = self.query('''
            SELECT
                codepoint,
                name,
                unicode(to_upper),
                unicode(fn1.unicode_upper(uchar)) 
            FROM utest.unicodedata
            WHERE codepoint in (1010)
                and (to_upper != fn1.unicode_upper(uchar))
                and (uchar != fn1.unicode_upper(uchar))
            ORDER BY codepoint
            LIMIT 50
            ''')
        self.assertRowsEqual([], rows)

    @requires('UNICODE_LEN')
    def test_unicode_len(self):
        rows = self.query('''
            SELECT codepoint, name
            FROM utest.unicodedata
            WHERE codepoint not between 55296 and 57343
                and len(uchar) != fn1.unicode_len(uchar)
            ORDER BY codepoint
            LIMIT 100
            ''')
        self.assertRowsEqual([], rows)

    @requires('UNICODE_LEN')
    #@expectedFailureIfLang('r')
    def test_unicode_len_on_undefined_block(self):
        '''DWA-19940 (R)'''
        if udf.pythonVersionInUdf == 2:
           rows = self.query('''
               SELECT codepoint, name
               FROM utest.unicodedata
               WHERE len(uchar) != fn1.unicode_len(uchar)
                   and codepoint between 55296 and 57343
               ORDER BY codepoint
               LIMIT 100
               ''')
           self.assertRowsEqual([], rows)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', help='connection string')
    parser.add_argument('--script-languages', help='definition of the SCRIPT_LANGUAGES variable')
    opts, _unknown = parser.parse_known_args()
    setattr(udf, 'pythonVersionInUdf', getPythonVersionInUDFs(opts.server, opts.script_languages))
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
