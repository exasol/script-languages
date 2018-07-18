#!/usr/opt/bs-python-2.7/bin/python

import datetime
import json
import os
import sys
import urllib

from textwrap import dedent

import pytz

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData

class CJSON(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')
   
        self.query(dedent('''\
                CREATE EXTERNAL SCALAR SCRIPT
                cjson_decode(json VARCHAR(10000))
                RETURNS VARCHAR(10000) AS
                # redirector @@redirector_url@@

                import json
                import cjson

                def run(ctx):
                    return json.dumps(cjson.decode(ctx.json))
                '''))
    
        self.query(dedent('''\
                CREATE EXTERNAL SCALAR SCRIPT
                cjson_encode(json VARCHAR(10000))
                RETURNS VARCHAR(10000) AS
                # redirector @@redirector_url@@

                import json
                import cjson

                def run(ctx):
                    return cjson.encode(json.loads(ctx.json))
                '''))
    
    def test_decode_empty_list(self):
        rows = self.query('SELECT cjson_decode(?) FROM DUAL', '[]')
        self.assertRowsEqual([('[]',)], rows)

    def test_encode_empty_list(self):
        rows = self.query('SELECT cjson_encode(?) FROM DUAL', '[]')
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
        rows = self.query('SELECT cjson_decode(?) FROM DUAL', data)
        self.assertRowsEqual([(data,)], rows)

    def test_encode_structured_data(self):
        data = json.dumps(self.nested())
        rows = self.query('SELECT cjson_encode(?) FROM DUAL', data)
        self.assertRowsEqual([(data,)], rows)


class Numpy(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')

    @useData((x,) for x in (3, 30, 300))
    def test_numpy_inverse(self, dim):
        self.query(dedent('''\
                CREATE EXTERNAL SCALAR SCRIPT
                numpy(dim INTEGER)
                RETURNS boolean AS
                # redirector @@redirector_url@@

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
                CREATE EXTERNAL SCALAR SCRIPT
                tz_convert_py(dt TIMESTAMP, tz VARCHAR(100))
                RETURNS TIMESTAMP AS
                # redirector @@redirector_url@@
        
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

