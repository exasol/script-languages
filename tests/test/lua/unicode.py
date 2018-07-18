#!/usr/opt/bs-python-2.7/bin/python

import locale
import os
import subprocess
import sys
import unicodedata

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import skip

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

class LuaUnicode(udf.TestCase):

    @classmethod
    def setUpClass(cls):
        sql = udf.fixindent('''
            DROP SCHEMA FN2 CASCADE;
            CREATE SCHEMA FN2;

            CREATE LUA SCALAR SCRIPT
            lua_match(c CHAR(1))
            RETURNS CHAR(2) AS
            magic_set = {}
            for c in string.gmatch("()%.+-*?[^$", ".") do
              magic_set[c] = true
            end
            function run(ctx)
                local c = ctx.c
              if (c ~= null) and (not magic_set[c]) then
                local txt = "x" .. c .. "x"
                return unicode.utf8.match(txt, c)
              end
            end
            /

            CREATE LUA SCALAR SCRIPT
            lua_gmatch(text VARCHAR(350))
            EMITS (w CHAR(1), c DOUBLE) AS
            function run(ctx)
                local txt = ctx.text
                if txt ~= null then
                    for c in unicode.utf8.gmatch(txt, ".") do
                    ctx.emit(c, 1)          
                    end     
                end
            end
            /

            CREATE LUA SCALAR SCRIPT
            lua_gsub(text VARCHAR(350))
            EMITS (w CHAR(1), c DOUBLE) AS
            function run(ctx)
               if ctx.text ~= null
               then
                   unicode.utf8.gsub(ctx.text, '.', function(w) ctx.emit(w,1) end)
               end
            end
            / 

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
    
            IMPORT INTO fn2.unicodedata
            FROM LOCAL CSV FILE '/share/fs8/Databases/UDF/unicode.csv'
            ROW SEPARATOR = 'CRLF'
            REJECT LIMIT 0;''')

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

    def setUp(self):
        '''Mixing different connections (JDBC/ODBC) may result in
        the ODBC connection to not see the commits of JDBC.
        Therefore, force a new transaction'''
        self.commit()
        self.query('OPEN SCHEMA fn2')

    def test_unicode_match(self):
        rows = self.query('''
            SELECT * FROM (
                SELECT codepoint, 
                       uchar, 
                       fn2.lua_match(uchar), 
                       uchar = fn2.lua_match(uchar) AS m 
                FROM fn2.unicodedata
            )
            WHERE (m IS NULL OR m = FALSE)
              AND codepoint not in (0,36,37,40,41,42,43,45,46,63,91,94)''')
        self.assertRowsEqual([], rows)


    @skip('manual test for DWA-13860, DWA-17091')
    def test_unicode_gmatch(self):
        nrows = self.query('''
                SELECT count(uchar) * 3
                FROM fn2.unicodedata
                WHERE codepoint BETWEEN 382976 AND 385152''')[0][0]
        for _ in range(25):
            self.query('''
                    SELECT fn2.lua_gmatch(uchar)
                    FROM (
                        SELECT 'x'||uchar||'x' AS uchar
                        FROM fn2.unicodedata
                        WHERE codepoint BETWEEN 382976 AND 385152
                    )''')
            self.assertEqual(nrows, self.rowcount())
    
    def test_unicode_gsub(self):
        rows = self.query('''
            SELECT unicode(w) FROM (
                SELECT fn2.lua_gsub(uchar) FROM fn2.unicodedata
            ) ORDER BY 1 ASC''')
        self.assertEqual(1114111, self.rowcount())
        s1 = set(range(1,1114112))
        s2 = set(x[0] for x in rows)
        self.assertEqual(s1 - s2, s2 - s1)


class LuaUnicodePattern(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA fn3 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA fn3')

    @skip('manual test for DWA-13860')
    def test_unicode_gmatch_classes(self):
        self.query(udf.fixindent('''
                CREATE lua SCALAR SCRIPT
                gmatch_pattern(w VARCHAR(1000))
                EMITS (w VARCHAR(1000)) AS

                function run(ctx)
                    local word = ctx.w
                    if word ~= null then
                        for i in unicode.utf8.gmatch(word, '([%w%p]+)') do
                            ctx.emit(i)
                        end
                    end
                end
                /
                '''))
        prefix = 0x1eba
        for u in range(sys.maxunicode):
            try:
                self.query('''
                    SELECT gmatch_pattern(
                                unicodechr(%d) || unicodechr(?))
                    FROM DUAL''' % prefix, u)
                #print u
            except:
                print 'U+%04X' %u, unicodedata.name(unichr(u), 'U+%04X' % u)


if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
