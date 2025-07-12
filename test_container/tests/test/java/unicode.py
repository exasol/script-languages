#!/usr/bin/env python3
# encoding: utf8

import locale
import os
import string
import subprocess

from exasol_python_test_framework import udf

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class JavaUnicode(udf.TestCase):
    def test_unicode_umlaute(self):
        cmd = '''%(exaplus)s -c %(conn)s -u sys -P exasol
		-no-config -autocommit ON -L -pipe -jdbcparam validateservercertificate=0''' % {
			'exaplus': os.environ.get('EXAPLUS',
				'/usr/opt/EXASuite-5/EXASolution-5.0.rc1/bin/Console/exaplus'),
			'conn': udf.opts.server
			}
        env = os.environ.copy()
        env['PATH'] = '/usr/opt/jdk1.8.0_latest/bin:' + env['PATH']
        env['LC_ALL'] = 'en_US.UTF-8'
        exaplus = subprocess.Popen(cmd.split(), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT)

        u = 'äöüß' + chr(382) + chr(65279) + chr(63882) + chr(64432)
        umlaute = 'äöüß'

        sql = udf.fixindent('''
            DROP SCHEMA fn1 CASCADE;
            CREATE SCHEMA fn1;
            CREATE OR REPLACE java SCALAR SCRIPT
            unicode_in_script_body()
            RETURNS INTEGER AS
            class UNICODE_IN_SCRIPT_BODY {
                static int run(ExaMetadata exa, ExaIterator ctx) {
                    String letters = "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
                    letters = letters.concat("%s");
                    int[] codePoints = {382, 65279, 63882, 64432};
                    letters = letters.concat(new String(codePoints, 0, codePoints.length));
                    return letters.codePointCount(0, letters.length());
                }
            }
            /
            SELECT 'x' || fn1.unicode_in_script_body() || 'x' FROM DUAL;
        ''' % umlaute)
        out, _err = exaplus.communicate(sql.encode('utf8'))
        expected = 'x%dx' % (len(string.ascii_letters) + len(u))
        self.assertIn(expected, out.decode('utf-8'))

if __name__ == '__main__':
    udf.main()
