#!/usr/bin/env python3
# encoding: utf8

import locale
import os
import string
import subprocess

from exasol_python_test_framework import udf

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class PythonUnicode(udf.TestCase):
    def test_unicode_umlaute(self):
        cmd = '''%(exaplus)s -c %(conn)s -u sys -P exasol
		-no-config -autocommit ON -L -pipe''' % {
            'exaplus': os.environ.get('EXAPLUS',
                                      '/usr/opt/EXASuite-4/EXASolution-4.2.9/bin/Console/exaplus'),
            'conn': udf.opts.server
        }
        env = os.environ.copy()
        env['PATH'] = '/usr/opt/jdk1.8.0_latest/bin:' + env['PATH']
        env['LC_ALL'] = 'en_US.UTF-8'
        exaplus = subprocess.Popen(cmd.split(), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)

        u = 'äöüß' + chr(382) + chr(65279) + chr(63882) + chr(64432)

        sql = udf.fixindent('''
            DROP SCHEMA fn1 CASCADE;
            CREATE SCHEMA fn1;
            CREATE OR REPLACE python3 SCALAR SCRIPT
            unicode_in_script_body()
            RETURNS INTEGER AS

            import string
            letters = string.ascii_letters + u'%s'
            

            def run(ctx):
                return len(letters)
            /
            SELECT 'x' || fn1.unicode_in_script_body() || 'x' FROM DUAL;
        ''' % u)
        out, _err = exaplus.communicate(sql.encode('utf-8'))
        expected = 'x%dx' % (len(string.ascii_letters) + len(u))
        self.assertIn(expected, out.decode('utf-8'))


if __name__ == '__main__':
    udf.main()
