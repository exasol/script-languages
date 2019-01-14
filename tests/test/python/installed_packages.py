#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys
import urllib

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

running_in_travis = 'TRAVIS' in os.environ

import udf
from exatest.servers import HTTPServer
from exatest.utils import tempdir
from exatest.testcase import skipIf

class PythonPackages(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA SCHEMA_PACKAGES CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA SCHEMA_PACKAGES')
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT "TESTREQ" (url varchar(1000))
            EMITS (the_get INT) AS

            import sys
            import glob

            sys.path.extend(glob.glob('/buckets/bfsdefault/default/*'))

            import requests
            def run(ctx):
                r = requests.get('http://'+ctx.url)
                ctx.emit(r.status_code)
            '''))

    @skipIf(running_in_travis, reason="This test is not supported when running in travis")
    def test_package_requests(self):
        with tempdir() as tmp:
            with open(os.path.join(tmp, 'foo.xml'), 'w') as f:
                f.write('''<foo/>\n''')
            with HTTPServer(tmp) as hs:
                querytext = 'select testreq(\'' + hs.address[0]+ ':' + format(hs.address[1])+ '\')'
                rows2 = self.query(querytext)
                self.assertEqual(200, rows2[0][0])

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
