#!/usr/bin/env python2.7
# encoding: utf8

import locale
import os
import string
import subprocess
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../../lib'))

import udf

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

        u = u'äöüß' + unichr(382) + unichr(65279) + unichr(63882) + unichr(64432)

        sql = udf.fixindent('''
            DROP SCHEMA fn1 CASCADE;
            CREATE SCHEMA fn1;
            CREATE OR REPLACE python SCALAR SCRIPT
            unicode_in_script_body()
            RETURNS INTEGER AS

            import string
            letters = string.ascii_letters + u'%s'
            

            def run(ctx):
                return len(letters)
            /
            SELECT 'x' || fn1.unicode_in_script_body() || 'x' FROM DUAL;
        ''' % u)
        out, _err = exaplus.communicate(sql.encode('utf8'))
        expected = 'x%dx' % (len(string.ascii_letters) + len(u))
        self.assertIn(expected, out)

if __name__ == '__main__':
    udf.main()
