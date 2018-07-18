#!/usr/opt/bs-python-2.7/bin/python
# encoding: utf8

import os
import sys
import datetime

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class CompatOldProtocol(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN22 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN22')
        self.query(udf.fixindent('''
                CREATE EXTERNAL SET SCRIPT DEFAULT_VAREMIT_GENERIC_EMIT (a varchar(100)) EMITS (...) AS
                # redirector @@redirector_url@@
                def run(ctx):
                    ctx.emit(*([ctx[0]]*exa.meta.output_column_count))
                def default_output_columns():
                    return "a varchar(100)"
                '''))

        
    def test_old_vmpython_chokes_on_new_message_types(self):
        with self.assertRaisesRegexp(Exception, r'Unexpected message'):
            self.query('''SELECT FN22.DEFAULT_VAREMIT_GENERIC_EMIT('Oh my!') FROM DUAL''')

        

if __name__ == '__main__':
    udf.main()
                
# vim: ts=4:sts=4:sw=4:et:fdm=indent
