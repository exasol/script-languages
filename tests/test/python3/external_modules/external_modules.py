#!/usr/bin/env python2.7

import datetime
import json
import os
import sys
import urllib

from textwrap import dedent

import pytz

sys.path.append(os.path.realpath(__file__ + '/../../../../lib'))

import udf
from udf import useData

class ExternalModulesImportTest(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')

    modules = '''
            ujson
            lxml
            numpy
            pytz
            scipy
            '''.split()
    @useData((x,) for x in modules)
    def test_3rdparty_modules_are_importable(self, module):
        sql = udf.fixindent('''
            CREATE OR REPLACE python3 SCALAR SCRIPT
            i()
            RETURNS VARCHAR(1024) AS
            import %(module)s
            def run(ctx):
                try:
                    return %(module)s.__file__
                except AttributeError:
                    return "(built-in)"
            ''')
        self.query(sql % {'module': module})
        self.query('SELECT i() FROM DUAL')


class UJSON(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')
   
        self.query(dedent('''\
                CREATE python3 SCALAR SCRIPT
                ujson_decode(json VARCHAR(10000))
                RETURNS VARCHAR(10000) AS
                import json
                import ujson
                def run(ctx):
                    return json.dumps(ujson.decode(ctx.json))
                '''))
    
        self.query(dedent('''\
                CREATE python3 SCALAR SCRIPT
                ujson_encode(json VARCHAR(10000))
                RETURNS VARCHAR(10000) AS

                import json
                import ujson

                def run(ctx):
                    return ujson.encode(json.loads(ctx.json))
                '''))
    
    def test_decode_empty_list(self):
        rows = self.query('SELECT ujson_decode(?) FROM DUAL', '[]')
        self.assertRowsEqual([('[]',)], rows)

    def test_encode_empty_list(self):
        rows = self.query('SELECT ujson_encode(?) FROM DUAL', '[]')
        self.assertRowsEqual([('[]',)], rows)

    @staticmethod
    def nested():
        return [
            [1, 2, 3.3, -4.5e10],
            {"a": "A", "b": "B"},
            [],
            {},
            {"a": [1,2,3,4], "x": ["a", "b", "c"]},
            False,
            True,
            None,
            ]

    def test_decode_structured_data(self):
        data = json.dumps(self.nested())
        rows = self.query('SELECT ujson_decode(?) FROM DUAL', data)
        self.assertRowsEqual([(data,)], rows)

    def test_encode_structured_data(self):
        data = json.dumps(self.nested()).replace(" ","")
        rows = self.query('SELECT ujson_encode(?) FROM DUAL', data)
        self.assertRowsEqual([(data,)], rows)


class Numpy(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')

    @useData((x,) for x in (3, 30, 300))
    def test_numpy_inverse(self, dim):
        self.query(dedent('''\
                CREATE python3 SCALAR SCRIPT
                numpy(dim INTEGER)
                RETURNS boolean AS

                from numpy import *
                from numpy.linalg import inv
                from numpy.random import seed, random_sample

                def run(ctx):
                    dim = ctx.dim
                    seed(12345678 * dim)
                    A = random_sample((dim, dim))
                    Ai = inv(A)
                    R = dot(A, Ai) - identity(dim)
                    return bool(-1e-12 <= R.min() <= R.max() <= 1e-12)
                '''))
        rows = self.query('SELECT numpy(?) FROM dual', dim)
        self.assertRowsEqual([(True,)], rows)


class Pytz(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')

    timezones = '''
            America/Manaus
            Asia/Katmandu
            Asia/Tokyo
            Asia/Yerevan
            Europe/Berlin
            '''.split()

    @useData((tz,) for tz in timezones)
    def test_convert(self, tz):
        self.query(dedent('''\
                CREATE python3 SCALAR SCRIPT
                tz_convert_py(dt TIMESTAMP, tz VARCHAR(100))
                RETURNS TIMESTAMP as
        
                import pytz
        
                def run(ctx):
                    tz = pytz.timezone(ctx.tz)
                    dt_utc = ctx.dt.replace(tzinfo=pytz.utc)
                    dt = dt_utc.astimezone(tz)
                    return dt.replace(tzinfo=None)
                '''))
        dt = datetime.datetime(2012, 4, 3, 23, 59, 0)
        rows = self.query('SELECT tz_convert_py(?, ?) FROM dual', (dt, tz))
        converted = dt.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(tz))
        self.assertRowsEqual([(converted.replace(tzinfo=None),)], rows)

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

