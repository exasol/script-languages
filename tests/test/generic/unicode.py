#!/usr/bin/env python2.7
# coding: utf-8

import csv
import locale
import logging
import os
import subprocess
import sys
import tempfile
import unicodedata

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import (
    requires,
    useData,
    expectedFailureIfLang,
    )

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def utf8encoder(row):
    return tuple(
            (x.encode('utf-8') if isinstance(x, unicode) else x)
             for x in row)

def setUpModule():
    log = logging.getLogger('unicodedata')

    log.info('generating unicodedata CSV')
    with tempfile.NamedTemporaryFile(prefix='unicode-', suffix='.csv') as csvfile:
        c = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        for i in xrange(sys.maxunicode+1):
            u = unichr(i)
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
                east_asian_width VARCHAR(2) ASCII,
                mirrored BOOLEAN,
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
    @expectedFailureIfLang('java')
    @expectedFailureIfLang('r')
    def test_unicode_lower_is_subset_of_Unicode520_part1(self):
        rows = self.query('''
            SELECT
                codepoint,
                name,
                unicode(to_lower),
                unicode(fn1.unicode_lower(uchar))
            FROM utest.unicodedata
            WHERE codepoint not between 55296 and 57343
                and codepoint not in (8486, 8490, 8491)
                and (to_lower != fn1.unicode_lower(uchar))
                and (uchar != fn1.unicode_lower(uchar))
            ORDER BY codepoint
            LIMIT 50
            ''')
        self.assertRowsEqual([], rows)

    @requires('UNICODE_LOWER')
    @expectedFailureIfLang('java')
    @expectedFailureIfLang('r')
    def test_unicode_lower_is_subset_of_Unicode520_part1_on_undefined_block(self):
        '''DWA-19940 (R)'''
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
    @expectedFailureIfLang('lua')
    def test_unicode_lower_is_subset_of_Unicode520_part2(self):
        '''DWA-13702 (Lua)'''
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
    @expectedFailureIfLang('java')
    @expectedFailureIfLang('r')
    def test_unicode_upper_is_subset_of_Unicode520_part1(self):
        rows = self.query('''
            SELECT
                codepoint,
                name,
                unicode(to_upper),
                unicode(fn1.unicode_upper(uchar)) 
            FROM utest.unicodedata
            WHERE codepoint not between 55296 and 57343
                and codepoint not in (181, 1010, 8126)
                and (to_upper != fn1.unicode_upper(uchar))
                and (uchar != fn1.unicode_upper(uchar))
            ORDER BY codepoint
            LIMIT 50
            ''')
        self.assertRowsEqual([], rows)

    @requires('UNICODE_UPPER')
    @expectedFailureIfLang('java')
    @expectedFailureIfLang('r')
    def test_unicode_upper_is_subset_of_Unicode520_part1_on_undefined_block(self):
        '''DWA-19940 (R)'''
        rows = self.query('''
            SELECT
                codepoint,
                name,
                unicode(to_upper),
                unicode(fn1.unicode_upper(uchar))
            FROM utest.unicodedata
            WHERE codepoint not in (181, 1010, 8126)
                and (to_upper != fn1.unicode_upper(uchar))
                and (uchar != fn1.unicode_upper(uchar))
                and codepoint between 55296 and 57343
            ORDER BY codepoint
            LIMIT 50
            ''')
        self.assertRowsEqual([], rows)

    @requires('UNICODE_UPPER')
    @expectedFailureIfLang('lua')
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
    @expectedFailureIfLang('r')
    def test_unicode_len_on_undefined_block(self):
        '''DWA-19940 (R)'''
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
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
