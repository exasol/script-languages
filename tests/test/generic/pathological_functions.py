#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import requires
from exatest.testcase import skipIf
running_in_travis = 'TRAVIS' in os.environ

class Test(udf.TestCase):

    @requires('SLEEP')
    def test_query_timeout(self):
        self.query('ALTER SESSION SET QUERY_TIMEOUT = 10')
        try:
            with self.assertRaisesRegexp(Exception, 'Successfully reconnected after query timeout'):
                self.query('SELECT fn1.sleep(100) FROM dual')
        finally:
            self.query('ALTER SESSION SET QUERY_TIMEOUT = 0')
        
    @requires('MEM_HOG')
    @skipIf(running_in_travis, reason="This test is not supported when running in travis")
    def test_kill_mem_hog(self):
        self.query('SELECT fn1.mem_hog(100) FROM dual')
        err_text = {
            'lua': 'Connection lost after session running out of memory.',
            'ext-python': 'MemoryError:',
            'python': '(Connection lost after system running out of memory)|(Query terminated because system running out of memory)',
            'r': 'Connection lost after system running out of memory.',
            'java': 'java.lang.OutOfMemoryError: Java heap space',
            }
        mb = int(9.2 * 1024*1024)
        with self.assertRaisesRegexp(Exception, err_text[udf.opts.lang]):
            if udf.opts.lang == "ext-python":
                mb = mb*1024*1024
            self.query('SELECT fn1.mem_hog(%d) FROM dual' % mb)

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

