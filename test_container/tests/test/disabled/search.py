#!/usr/opt/bs-python-2.7/bin/python
# coding: UTF-8
'''
This is an annotated test file of the UDF testing framework. You can
execute it directly.
The only mandatory option is --server.

The framework is based on Python's unittest framework, so Python's 
documentation applies here as well.
'''

        cmd = '''%(exaplus)s -c %(conn)s -u sys -P exasol
		-no-config -autocommit ON -L -pipe -jdbcparam validateservercertificate=0''' % {
			'exaplus': os.environ.get('EXAPLUS',
				'/usr/opt/EXASuite-4/EXASolution-4.2.9/bin/Console/exaplus'),
			'conn': udf.opts.server
			}
        env = os.environ.copy()
        env['PATH'] = '/usr/opt/jdk1.8.0_latest/bin:' + env['PATH']
        exaplus = subprocess.Popen(cmd.split(), env=env, stdin=subprocess.PIPE)
        exaplus.communicate(udf.fixindent(sql))
        if exaplus.returncode != 0:
            logging.getLogger(cls.__class__.__name__).critical('EXAplus error')


    # @classmethod
    # def tearDownClass(cls):
    #     sql = '''drop schema ir_test cascade;'''
    #     cmd = '''/usr/opt/EXASuite-4/EXASolution-4.2.9/bin/Console/exaplus
    #         -c %s -u sys -P exasol -no-config -autocommit ON -L -pipe''' % udf.opts.server
    #     env = os.environ.copy()
    #     env['PATH'] = '/usr/opt/jdk1.8.0_latest/bin:' + env['PATH']
    #     exaplus = subprocess.Popen(cmd.split(), env=env, stdin=subprocess.PIPE)
    #     exaplus.communicate(sql)
    #     if exaplus.returncode != 0:
    #         logging.getLogger(cls.__class__.__name__).critical('EXAplus error')


    def setUp(self):
        self.query('open schema ir_test;')


    def tearDown(self):
        '''executed after each test'''
        pass

    def some_helper_function(self):
        '''not a test because it's name does not start with "test"'''
        pass

    def test_some_test_case(self):
        self.assertTrue(4, 2+2)
        self.assertLessEqual(2, 3)
        self.assertIn(4, [1,2,3,4])
        self.assertIsNotNone(42)
    
    def test_catched_exceptions(self):
        with self.assertRaises(IndexError):
            a = []
            print a[10]

        # check for exception texts:
        with self.assertRaisesRegex(Exception, 'FooBar.*Error'):
            raise ValueError('FooBarBazError')

    def test_case_with_comment(self):
        '''Test case can have a comment

        The first line is printed with --verbose
        '''
        pass

    def test_query_result(self):
        rows = self.query(udf.fixindent('''execute script search_for('perl',1);'''))
        self.assertRowsEqual([("/usr/share/doc/perl-IO-Socket-SSL-1.01/docs/debugging.txt (0.022727272727273)",)], rows)



if __name__ == '__main__':
    udf.main()

