#!/usr/opt/bs-python-2.7/bin/python

import locale
import os
import subprocess
import sys
import traceback

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import (
        expectedFailureIfLang,
        requires,
        SkipTest,
        timer,
        )

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

class WordCount(udf.TestCase):

    def setUp(self):
        self.query('OPEN SCHEMA FN1')

    @requires('PERFORMANCE_MAP_WORDS')
    @requires('PERFORMANCE_REDUCE_COUNTS')
    def test_word_count(self):
        sql = '''
        SELECT COUNT(*) FROM (
            SELECT performance_reduce_counts(w, c)
            FROM (
	            SELECT performance_map_words(varchar02)
	            FROM test.enginetablebig1
            )
            GROUP BY w
            ORDER BY 1 DESC)'''

        with timer() as t:
            ret = self.query(sql)
        print "test_word_count query:", t.duration, repr(ret)
        self.assertLessEqual(t.duration, 160)

    @requires('PERFORMANCE_MAP_WORDS')
    @requires('PERFORMANCE_REDUCE_COUNTS_FAST0')
    def test_word_count_fast0(self):
        sql = '''
        SELECT COUNT(*) FROM (
            SELECT performance_reduce_counts_fast0(w, c)
            FROM (
	            SELECT performance_map_words(varchar02)
	            FROM test.enginetablebig1
            )
            GROUP BY w
            ORDER BY 1 DESC)'''

        with timer() as t:
            ret = self.query(sql)
        print "test_word_count_fast0 query:", t.duration, repr(ret)
        self.assertLessEqual(t.duration, 160)

    @requires('PERFORMANCE_MAP_WORDS')
    @requires('PERFORMANCE_REDUCE_COUNTS_FAST7')
    def test_word_count_fast7(self):
        sql = '''
        SELECT COUNT(*) FROM (
            SELECT performance_reduce_counts_fast7(w, c)
            FROM (
	            SELECT performance_map_words(varchar02)
	            FROM test.enginetablebig1
            )
            GROUP BY w
            ORDER BY 1 DESC)'''

        with timer() as t:
            ret = self.query(sql)
        print "test_word_count_fast7 query:", t.duration, repr(ret)
        self.assertLessEqual(t.duration, 160)

    @requires('PERFORMANCE_MAP_WORDS')
    @requires('PERFORMANCE_REDUCE_COUNTS_FAST77')
    def test_word_count_fast77(self):
        sql = '''
        SELECT COUNT(*) FROM (
            SELECT performance_reduce_counts_fast77(w, c)
            FROM (
	            SELECT performance_map_words(varchar02)
	            FROM test.enginetablebig1
            )
            GROUP BY w
            ORDER BY 1 DESC)'''

        with timer() as t:
            ret = self.query(sql)
        print "test_word_count_fast77 query:", t.duration, repr(ret)
        self.assertLessEqual(t.duration, 160)

    @requires('PERFORMANCE_MAP_WORDS')
    @requires('PERFORMANCE_REDUCE_COUNTS_FAST777')
    def test_word_count_fast777(self):
        sql = '''
        SELECT COUNT(*) FROM (
            SELECT performance_reduce_counts_fast777(w, c)
            FROM (
	            SELECT performance_map_words(varchar02)
	            FROM test.enginetablebig1
            )
            GROUP BY w
            ORDER BY 1 DESC)'''

        with timer() as t:
            ret = self.query(sql)
        print "test_word_count_fast777 query:", t.duration, repr(ret)
        self.assertLessEqual(t.duration, 160)

    @requires('PERFORMANCE_MAP_WORDS')
    @requires('PERFORMANCE_REDUCE_COUNTS_FAST7777')
    def test_word_count_fast7777(self):
        sql = '''
        SELECT COUNT(*) FROM (
            SELECT performance_reduce_counts_fast7777(w, c)
            FROM (
	            SELECT performance_map_words(varchar02)
	            FROM test.enginetablebig1
            )
            GROUP BY w
            ORDER BY 1 DESC)'''

        with timer() as t:
            ret = self.query(sql)
        print "test_word_count_fast7777 query:", t.duration, repr(ret)
        self.assertLessEqual(t.duration, 160)

    @requires('PERFORMANCE_MAP_WORDS')
    @requires('PERFORMANCE_REDUCE_COUNTS_FAST777777')
    def test_word_count_fast777777(self):
        sql = '''
        SELECT COUNT(*) FROM (
            SELECT performance_reduce_counts_fast777777(w, c)
            FROM (
	            SELECT performance_map_words(varchar02)
	            FROM test.enginetablebig1
            )
            GROUP BY w
            ORDER BY 1 DESC)'''

        with timer() as t:
            ret = self.query(sql)
        print "test_word_count_fast777777 query:", t.duration, repr(ret)
        self.assertLessEqual(t.duration, 160)

    @requires('PERFORMANCE_MAP_WORDS')
    @requires('PERFORMANCE_REDUCE_COUNTS_FAST77777777')
    def test_word_count_fast77777777(self):
        sql = '''
        SELECT COUNT(*) FROM (
            SELECT performance_reduce_counts_fast77777777(w, c)
            FROM (
	            SELECT performance_map_words(varchar02)
	            FROM test.enginetablebig1
            )
            GROUP BY w
            ORDER BY 1 DESC)'''

        with timer() as t:
            ret = self.query(sql)
        print "test_word_count_fast77777777 query:", t.duration, repr(ret)
        self.assertLessEqual(t.duration, 160)

    @requires('PERFORMANCE_MAP_UNICODE_WORDS')
    @requires('PERFORMANCE_REDUCE_COUNTS')
    @expectedFailureIfLang('lua')
    def test_word_unicode_count(self):
        '''DWA-13860 (lua)'''
        sql = '''
            SELECT performance_reduce_counts(w, c)
            FROM (
	            SELECT performance_map_unicode_words(c3_varchar100)
	            FROM test.enginetablebigunicode
            )
            GROUP BY w
            ORDER BY 1 DESC'''

        with timer() as t:
            self.query(sql)
        self.assertLessEqual(t.duration, 11)

class FrequencyAnalysis(udf.TestCase):

    maxDiff = 1024 * 20

    def compare(self, old, new):
        self.log.info('compare new data with reference data')
        n_old = len(list(old))
        n_new = len(list(new))
        self.log.info('old data has %d lines', n_old)
        self.log.info('new data has %d lines', n_new)
        if max(n_old, n_new) <= 50:
            self.assertEqual(old, new)
        else:
            self.log.info('switching to compact comparison')
            old_set = set(old)
            new_set = set(new)
            only_new = list(sorted(new_set.difference(old_set)))
            only_old = list(sorted(old_set.difference(new_set)))
            if max(len(only_new), len(only_old)) <= 200:
                self.assertEqual(([], []), (only_old, only_new))
            else:
                self.log.info('diff is still to big')
                self.fail("difference: +%d/-%d elements" %
                        (len(only_new), len(only_old)))

    @classmethod
    def setUpClass(cls):
        sql = """
            DROP SCHEMA daten CASCADE;
            CREATE SCHEMA daten;

            CREATE TABLE wiki_freq(w char(1), c INTEGER);

            IMPORT INTO wiki_freq
            FROM LOCAL CSV FILE '/share/fs8/Databases/UDF/freebase-frequency_analysis.csv'
            COLUMN SEPARATOR = 'TAB'
            REJECT LIMIT 0;

            CREATE TABLE wiki_names(id INTEGER IDENTITY PRIMARY KEY, text VARCHAR(350));

            IMPORT INTO wiki_names(text)
            FROM LOCAL CSV FILE '/share/fs8/Databases/UDF/freebase-export.csv'
            COLUMN SEPARATOR = 'TAB'
            REJECT LIMIT 0;"""

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
                stderr=subprocess.STDOUT,
                )
        out, _err = exaplus.communicate(sql)
        if exaplus.returncode != 0:
            cls.log.critical('EXAplus error: %d', exaplus.returncode)
            cls.log.error(out)
        else:
            cls.log.debug(out)

    @requires('PERFORMANCE_REDUCE_CHARACTERS')
    @requires('PERFORMANCE_MAP_CHARACTERS')
    def test_frequency_analysis(self):
        if udf.opts.lang == 'r':
            raise SkipTest('this R implementation is too slow')
        with timer() as t1:
            rows1 = self.query('''
            SELECT fn1.performance_reduce_characters(w, c)
            FROM (
                SELECT fn1.performance_map_characters(text)
                FROM daten.wiki_names
            )
            GROUP BY w
            ORDER BY c DESC, w ASC''')

        with timer() as t2:
            rows2 = self.query('''
            SELECT w, c 
            FROM daten.wiki_freq
            ORDER BY c DESC, w ASC''')

        data = [tuple(x) for x in rows1]
        reference = [tuple(x) for x in rows2]
        print "test_frequency_analysis query:", t1.duration, t2.duration
        self.compare(reference, data)

    @requires('PERFORMANCE_REDUCE_CHARACTERS_FAST')
    @requires('PERFORMANCE_MAP_CHARACTERS_FAST')
    def test_frequency_analysis_fast(self):
        with timer() as t1:
            rows1 = self.query('''
            SELECT fn1.performance_reduce_characters_fast(w, c)
            FROM (
                SELECT fn1.performance_map_characters_fast(text)
                FROM daten.wiki_names
                WHERE text IS NOT NULL
            )
            GROUP BY w
            ORDER BY c DESC, w ASC''')

        with timer() as t2:
            rows2 = self.query('''
            SELECT w, c 
            FROM daten.wiki_freq
            ORDER BY c DESC, w ASC''')

        data = [tuple(x) for x in rows1]
        reference = [tuple(x) for x in rows2]
        print "test_frequency_analysis_fast query:", t1.duration, t2.duration
        self.compare(reference, data)

    @requires('PERFORMANCE_REDUCE_CHARACTERS_FAST')
    @requires('PERFORMANCE_MAP_CHARACTERS_FAST0')
    def test_frequency_analysis_fast0(self):
        with timer() as t1:
            rows1 = self.query('''
            SELECT fn1.performance_reduce_characters_fast(w, c)
            FROM (
                SELECT fn1.performance_map_characters_fast0(text)
                FROM daten.wiki_names
                WHERE text IS NOT NULL
            )
            GROUP BY w
            ORDER BY c DESC, w ASC''')

        with timer() as t2:
            rows2 = self.query('''
            SELECT w, c 
            FROM daten.wiki_freq
            ORDER BY c DESC, w ASC''')

        data = [tuple(x) for x in rows1]
        reference = [tuple(x) for x in rows2]
        print "test_frequency_analysis_fast0 query:", t1.duration, t2.duration
        self.compare(reference, data)

    @requires('PERFORMANCE_REDUCE_CHARACTERS')
    @requires('PERFORMANCE_MAP_CHARACTERS')
    def test_frequency_analysis_light(self):
        if udf.opts.lang != 'r':
            raise SkipTest('light test for R only')
        self.query('''
            SELECT fn1.performance_reduce_characters(w, c)
            FROM (
                SELECT fn1.performance_map_characters(text)
                FROM daten.wiki_names
                WHERE mod(length(daten.wiki_names.text), 50) = 2
            )
            GROUP BY w
            ORDER BY c DESC, w ASC''')

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
