#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class AvailablePythonPackages(udf.TestCase):
    def setUp(self): 
        self.query('create schema available_packages', ignore_errors=True) 



    data = [
            ("azure.batch",), 
            ("azure.cosmos",), 
            ("azure.eventgrid",)
            ("azure.eventhub",)
            ("azure.eventhub.checkpointstoreblob",)
            ("azure.eventhub.checkpointstoreblob.aio",)
            ("azure.identity",)
            ("azure.keyvault",)
            ("azure.keyvault.certificates",)
            ("azure.keyvault.keys",)
            ("azure.keyvault.secrets",)
            ("azure.kusto-data",)
            ("azure.loganalytics",)
            ("azure.servicebus",)
            ("azure.storage.blob",)
            ("azure.storage.file.datalake",)
            ("azure.storage.file.share",)
            ("azure.storage.queue",)
        ]

    @useData(data)
    def import_test(self, pkg, fail=False, alternative=None):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT available_packages.test_import_of_package() returns int AS
            import %s
            def run(ctx): return 1
            /
            ''' % (pkg)))
        try:
            rows = self.query('''SELECT available_packages.test_import_of_package() FROM dual''')
            if not fail:
                self.assertRowsEqual([(1,)], rows)
            else:
                assert 'Expected Failure' == 'not found'
        except:
            if fail:
                return
            if alternative:
                self.import_test(alternative,fail)
            else:
                raise





if __name__ == '__main__':
    udf.main()

