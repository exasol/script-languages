#!/usr/bin/env python3

import locale
import os
import subprocess

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import timer, SkipTest, skip

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class WordCount(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        # Create Python3 UDFs for performance testing
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT PERFORMANCE_MAP_WORDS(w VARCHAR(1000))
            EMITS (w VARCHAR(1000), c INTEGER) AS
            
            import re
            import string
            
            pattern = re.compile(r'''+'\'\'\'([]\w!"#$%&\\\'()*+,./:;<=>?@[\\\\^_`{|}~-]+)\'\'\''+''')
            
            def run(ctx):
                if ctx.w is not None:
                    for w in re.findall(pattern, ctx.w):
                        ctx.emit(w, 1)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT PERFORMANCE_MAP_UNICODE_WORDS(w VARCHAR(1000))
            EMITS (w VARCHAR(1000), c INTEGER) AS
            
            import re
            import string
            
            pattern = re.compile(r'''+'\'\'\'([]\w!"#$%&\\\'()*+,./:;<=>?@[\\\\^_`{|}~-]+)\'\'\''+''', re.UNICODE)
            
            def run(ctx):
                if ctx.w is not None:
                    for w in re.findall(pattern, ctx.w):
                        ctx.emit(w, 1)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT PERFORMANCE_REDUCE_COUNTS(w VARCHAR(1000), c INTEGER)
            EMITS (w VARCHAR(1000), c INTEGER) AS
            
            def run(ctx):
                word = ctx.w
                count = 0
                while True:
                    count += ctx.c
                    if not ctx.next(): break
                ctx.emit(word, count)
            /
        '''))

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
        print("test_word_count query:", t.duration, repr(ret))
        self.assertLessEqual(t.duration, 160)

    def test_word_unicode_count(self):
        """Test Unicode word counting"""
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


@skip('csv data for tables wiki_freq and wiki_names is currently not available')
class FrequencyAnalysis(udf.TestCase):
    maxDiff = 1024 * 20

    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        # Create Python3 UDFs for character frequency analysis
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT PERFORMANCE_MAP_CHARACTERS(text VARCHAR(1000))
            EMITS (w CHAR(1), c INTEGER) AS
            def run(ctx):
                if ctx.text is not None:
                    for c in ctx.text:
                        ctx.emit(c, 1)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT PERFORMANCE_REDUCE_CHARACTERS(w CHAR(1), c INTEGER)
            EMITS (w CHAR(1), c INTEGER) AS
            
            def run(ctx):
                c = 0
                w = ctx.w
                if w is not None:
                    while True:
                        c += 1
                        if not ctx.next(): break
                    ctx.emit(w, c)
            /
        '''))

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
		        -no-config -autocommit ON -L -pipe -jdbcparam validateservercertificate=0''' % {
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
        out, _err = exaplus.communicate(sql.encode('utf-8'))
        if exaplus.returncode != 0:
            cls.log.critical('EXAplus error: %d', exaplus.returncode)
            cls.log.error(out)
        else:
            cls.log.debug(out)

    def test_frequency_analysis(self):
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
        print("test_frequency_analysis query:", t1.duration, t2.duration)
        self.compare(reference, data)

    def test_frequency_analysis_light(self):
        """Lighter version processing subset of data"""
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
