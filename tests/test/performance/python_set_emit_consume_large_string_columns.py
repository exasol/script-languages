#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys
import time
import tempfile
import subprocess
import threading

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))
sys.path.append(os.path.realpath(__file__ + '/..'))

import udf
from abstract_performance_test import AbstractPerformanceTest

class SetEmitConsumeLargeStringColumnPythonPeformanceTest(AbstractPerformanceTest):

    def generate_data(self, multiplier, base=10):
#        self.number_of_characters = 2000000
        self.number_of_characters = 1864129
        columns_definition = ",".join(["column%s VARCHAR(%s)"%(i,self.number_of_characters) for i in range(self.number_of_columns)])
        self.column_names = ",".join(["column%s"%(i) for i in range(self.number_of_columns)])

        create_table_sql = 'CREATE OR REPLACE TABLE BATCH (%s);'%columns_definition
        column_values = ",".join(["a"*self.number_of_characters for i in range(self.number_of_columns)])
        tmpdir = tempfile.mkdtemp()
        fifo_filename = os.path.join(tmpdir, 'myfifo')
        import_table = '''IMPORT into BATCH from local CSV file '%s';'''%fifo_filename
        try:
            os.mkfifo(fifo_filename)
            cmd = '''%(exaplus)s -c %(conn)s -u %(user)s -P %(password)s -s %(schema)s
                            -no-config -autocommit ON -L -pipe''' % {
                                    'exaplus': os.environ.get('EXAPLUS'),
                                    'conn': udf.opts.server,
                                    'user': self.user,
                                    'password': self.password,
                                    'schema': self.schema
                                    }
            env = os.environ.copy()
            env['LC_ALL'] = 'en_US.UTF-8'
            exaplus = subprocess.Popen(
                        cmd.split(), 
                        env=env, 
                        stdin=subprocess.PIPE, 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
            write_trhead = threading.Thread(target=self.write_into_fifo, args=(fifo_filename, column_values, base))
            write_trhead.start()
            sql=create_table_sql+"\n"+import_table+"\n"+"commit;"
            out, _err = exaplus.communicate(sql.encode('utf8'))
            print(out)
            print(_err)
            write_trhead.join()
        finally:
            os.remove(fifo_filename)
            os.rmdir(tmpdir)
        create_table_sql = 'CREATE OR REPLACE TABLE T (%s);'%columns_definition
        self.query(create_table_sql)
        for i in range(multiplier):
            self.query('''INSERT INTO T select * from BATCH;''')
        self.query("commit")


    def write_into_fifo(self, fifo_filename, column_values, rows):
        with open(fifo_filename,"w") as f:
            for i in range(rows):
                f.write(column_values)
                f.write("\n")



    def setUp(self):
        self.create_schema();
        self.number_of_columns = 9
        self.query(udf.fixindent('''
                CREATE PYTHON SET SCRIPT CONSUME_NEXT_COLUMNS(...) RETURNS INT AS
                def run(ctx):
                    count = 0
#                    while(ctx.next()):
                    for i in range(%s):
                        stringVal = ctx[0]
#                    ctx.emit(count)
                    return count
                '''% self.number_of_columns))
        self.query("commit")
        self.generate_data(10)
        
    def tearDown(self):
        self.cleanup(self.schema)
    
    def test_consume_next_columns(self):
        self.run_test(15, 2.0, "SELECT CONSUME_NEXT_COLUMNS(%s) FROM T"%self.column_names)
        #self.run_test(2, 2.0, "SELECT %s FROM T"%self.column_names)

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

