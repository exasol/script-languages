#!/usr/opt/bs-python-2.7/bin/python

import os
import sys
import datetime

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

from exatest.servers import FTPServer
from exatest.utils import tempdir
import pyftpdlib.ftpserver as ftpserver

class NaNAndNOTNULL(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE OR REPLACE TABLE T(x DOUBLE CONSTRAINT nn NOT NULL)')

    def test_insert_nan_in_notnull_column(self):
        with self.assertRaisesRegexp(Exception, 'constraint violation'):
            self.query('''INSERT INTO T VALUES ?''', float('nan'))
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())

class InsertNULLinNOTNULLColumn(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_int32(self):
        self.query('CREATE OR REPLACE TABLE T(x DECIMAL(9,0) CONSTRAINT nn NOT NULL)')
        self.query('''INSERT INTO T VALUES ?''', 128)
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(128, rows[0][0])
        self.query('CREATE OR REPLACE TABLE T(x DECIMAL(9,0) CONSTRAINT nn NOT NULL)')
        self.query('''INSERT INTO T VALUES ?''', 32768)
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(32768, rows[0][0])
        self.query('CREATE OR REPLACE TABLE T(x DECIMAL(9,0) CONSTRAINT nn NOT NULL)')
        with self.assertRaisesRegexp(Exception, 'numeric value out of range'):
            self.query('''INSERT INTO T VALUES ?''', '2147483648')
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())

    def test_int64(self):
        self.query('CREATE OR REPLACE TABLE T(x DECIMAL(18,0) CONSTRAINT nn NOT NULL)')
        with self.assertRaisesRegexp(Exception, 'numeric value out of range'):
            self.query('''INSERT INTO T VALUES ?''', '9223372036854775808')
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())
    
    def test_int128(self):
        self.query('CREATE OR REPLACE TABLE T(x DECIMAL(36,0) CONSTRAINT nn NOT NULL)')
        with self.assertRaisesRegexp(Exception, 'numeric value out of range'):
            self.query('''INSERT INTO T VALUES ?''', '170141183460469231731687303715884105728')
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())

    def test_date(self):
        self.query('CREATE OR REPLACE TABLE T(x DATE CONSTRAINT nn NOT NULL)')
        with self.assertRaisesRegexp(Exception, 'year is out of range'):
            self.query('''INSERT INTO T VALUES ?''', datetime.date(0,0,0))
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())
    
    def test_timestamp(self):
        self.query('CREATE OR REPLACE TABLE T(x TIMESTAMP CONSTRAINT nn NOT NULL)')
        with self.assertRaisesRegexp(Exception, 'year is out of range'):
            self.query('''INSERT INTO T VALUES ?''', datetime.datetime(0,0,0,0,0,0,0))
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())

    def test_interval_ym(self):
        self.query('CREATE OR REPLACE TABLE T(x INTERVAL YEAR(9) TO MONTH CONSTRAINT nn NOT NULL)')
        with self.assertRaisesRegexp(Exception, 'leading precision of the interval is too small'):
            self.query('''INSERT INTO T VALUES ?''', '768614336404564650-8')
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())

    def test_bool(self):
        self.query('CREATE OR REPLACE TABLE T(x BOOLEAN CONSTRAINT nn NOT NULL)')
        with self.assertRaisesRegexp(Exception, 'Numeric value has to be 0 or 1'):
            self.query('''INSERT INTO T VALUES ?''', 128)
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())

    def test_varchar_emptystring(self):
        self.query('CREATE OR REPLACE TABLE T(x VARCHAR(10) CONSTRAINT nn NOT NULL)')
        with self.assertRaisesRegexp(Exception, 'constraint violation'):
            self.query('''INSERT INTO T VALUES ?''', '')
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())

    def test_char_emptystring(self):
        self.query('CREATE OR REPLACE TABLE T(x CHAR(10) CONSTRAINT nn NOT NULL)')
        with self.assertRaisesRegexp(Exception, 'constraint violation'):
            self.query('''INSERT INTO T VALUES ?''', '')
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())

    def test_varchar_nullbyte(self):
        self.query('CREATE OR REPLACE TABLE T(x VARCHAR(10) CONSTRAINT nn NOT NULL)')
        with self.assertRaisesRegexp(Exception, 'constraint violation'):
            self.query('''INSERT INTO T VALUES ?''', '\x00')
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())

    def test_char_nullbyte(self):
        self.query('CREATE OR REPLACE TABLE T(x CHAR(10) CONSTRAINT nn NOT NULL)')
        with self.assertRaisesRegexp(Exception, 'constraint violation'):
            self.query('''INSERT INTO T VALUES ?''', '\x00')
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())

    def test_geo(self):
        self.query('CREATE OR REPLACE TABLE T(x GEOMETRY CONSTRAINT nn NOT NULL)')
        with self.assertRaisesRegexp(Exception, 'constraint violation'):
            self.query('''INSERT INTO T VALUES ?''', '')
        rows = self.query('''SELECT x FROM T''')
        self.assertEqual(0, self.rowcount())


class ImportNULLinNOTNULLColumn(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def importHelper(self, csv, tableDefinition, exception):
        with tempdir() as tmp:
            with open(os.path.join(tmp, 'data.csv'), 'w') as f:
                f.write(csv)
            with FTPServer(tmp) as ftpd:
                url = 'ftp://anonymous:guest@%s:%d' % ftpd.address
                self.query('''
                        create connection ftpconnection to '%s'
                        ''' % url)
                self.query(tableDefinition)
                with self.assertRaisesRegexp(Exception, exception):
                    self.query('''
                    import into t from csv at ftpConnection file 'data.csv';
                               ''')
                rows = self.query('''SELECT * FROM T''')
                self.assertEqual(0, self.rowcount())

    def test_varchar_nullbyte(self):
        csv = '1,2\n3,\x00'
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x VARCHAR(10) NOT NULL, y varchar(10) not null)',
                          'constraint violation')

    def test_char_nullbyte(self):
        csv = '1,2\n3,\x00'
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x CHAR(10) NOT NULL, y char(10) not null)',
                          'constraint violation')

    def test_geo_nullbyte(self):
        csv = '"POINT(1 1)","POINT(1 1)"\n"POINT(1 1)",\x00'
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x GEOMETRY NOT NULL, y GEOMETRY not null)',
                          'Expected word but encountered number')

    def test_varchar_empty(self):
        csv = '1,2\n3,'
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x VARCHAR(10) NOT NULL, y varchar(10) not null)',
                          'constraint violation')

    def test_char_empty(self):
        csv = '1,2\n3,'
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x CHAR(10) NOT NULL, y char(10) not null)',
                          'constraint violation')

    def test_geo_empty(self):
        csv = '"POINT(1 1)","POINT(1 1)"\n"POINT(1 1)",""'
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x GEOMETRY NOT NULL, y GEOMETRY not null)',
                          'constraint violation')

    def test_date(self):
        csv = '{0},{1}\n{2},{3}'.format(datetime.date(1,1,1),datetime.date(2,2,2),datetime.date(3,3,3),'0000-00-00')
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x DATE CONSTRAINT nn NOT NULL, y DATE CONSTRAINT nn1 NOT NULL)',
                          'invalid value for YYYY format token')

    def test_timestamp(self):
        csv = '{0},{1}\n{2},{3}'.format(datetime.datetime(1,1,1,0,0,0,0), datetime.datetime(1,1,1,0,0,0,0), datetime.datetime(1,1,1,0,0,0,0), '0000-00-00 00:00:00')
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x TIMESTAMP CONSTRAINT nn NOT NULL, y TIMESTAMP CONSTRAINT nn1 NOT NULL)',
                          'invalid value for YYYY format token')

    def test_interval_ym(self):
        csv = '{0},{1}\n{2},{3}'.format('1-8','1-8','1-8','768614336404564650-8')
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x INTERVAL YEAR(9) TO MONTH CONSTRAINT nn NOT NULL, y INTERVAL YEAR(9) TO MONTH CONSTRAINT nn1 NOT NULL)',
                          'leading precision of the interval is too small')

    def test_bool(self):
        csv = '{0},{1}\n{2},{3}'.format('1','1','1','128')
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x BOOLEAN CONSTRAINT nn NOT NULL, y BOOLEAN CONSTRAINT nn1 NOT NULL)',
                          'Neither TRUE nor FALSE at string to boolean cast')

    def test_int64(self):
        csv = '{0},{1}\n{2},{3}'.format('1','1','1','9223372036854775808')
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x DECIMAL(18,0) CONSTRAINT nn NOT NULL, y DECIMAL(18,0) CONSTRAINT nn1 NOT NULL)',
                          'numeric value out of range')

    def test_int128(self):
        csv = '{0},{1}\n{2},{3}'.format('1','1','1','170141183460469231731687303715884105728')
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x DECIMAL(36,0) CONSTRAINT nn NOT NULL, y DECIMAL(36,0) CONSTRAINT nn1 NOT NULL)',
                          'numeric value out of range')

    def test_int32(self):
        csv = '{0},{1}\n{2},{3}'.format('1','1','1','2147483648')
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x DECIMAL(9,0) CONSTRAINT nn NOT NULL, y DECIMAL(9,0) CONSTRAINT nn1 NOT NULL)',
                          'numeric value out of range')

    def test_double(self):
        csv = '{0},{1}\n{2},{3}'.format('1','1','1',float('nan'))
        self.importHelper(csv,
                          'CREATE OR REPLACE TABLE T(x DOUBLE CONSTRAINT nn NOT NULL, y DOUBLE CONSTRAINT nn1 NOT NULL)',
                          'invalid character value for cast')


if __name__ == '__main__':
    udf.main()
