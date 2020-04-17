
import functools
import logging
import os
import re
import sys
import unittest
import subprocess

import pyodbc
from decimal import Decimal
from datetime import date
from datetime import datetime


import exatest
from exatest import *

from exatest.clients.odbc import ODBCClient, getScriptLanguagesFromArgs

capabilities = []
opts = None

# 
#   TestCase decorators
#
def requires(req):
    '''Skip test if requirements are not met (class method decorator)

    Unlike skipIf decorator, condition is evaluated at method run time.

    Usage:
    class Test(object):

    @requires('foo')
    def test_foo(self):
        # executed

    @requires('bar')
    def test_bar(self):
        # skipped
    '''
    def dec(func):
        @functools.wraps(func)
        def wrapper(*args):
            if not opts.lang:
                raise TypeError('"@requires" is only allowed for generic tests')
            if req not in capabilities:
                raise exatest.SkipTest('requires: %s' % req) 
            return func(*args)
        wrapper._sort_key = exatest.get_sort_key(func)
        return wrapper
    return dec

def expectedFailureIfLang(lang):
    '''Expect test to fail if lang is opts.lang'''
    def dec(func):
        @functools.wraps(func)
        def wrapper(*args):
            if lang == opts.lang:
                return unittest.expectedFailure(func)(*args)
            else:
                return func(*args)
        wrapper._sort_key = exatest.get_sort_key(func)
        return wrapper
    return dec

def fixindent(query):
    '''Remove indent from multi-line query text.

    SQL does not care about indent, but embedded languages (like Python)
    might. Use indent of first indented non-empty line as reference.

    Usage:

        sql = fixindent("""
                CREATE python SCALAR SCRIPT
                foo RETURNS int AS

                def run(context):
                    ...
                """)
    '''
    lines = query.split('\n')
    ref = ''
    indent = re.compile('^(\s+)\S+')
    for line in lines:
        match = indent.match(line)
        if match:
            ref = match.group(1)
            break
    if ref:
        return '\n'.join([
            (line.replace(ref, '', 1) if line.startswith(ref) else line)
            for line in lines])
    else:
        return query

class TestProgram(exatest.TestProgram):
    logger_name = 'udf.main'

    def parser_hook(self, parser):
        new_opts = parser.add_argument_group('UDF specific')
        new_opts.add_argument('--lang',
            help='programming language (default: %(default)s)')
        new_opts.add_argument('--redirector-url',
            help='comma separated list of redirector urls for external script service (default: %(default)s)')
        new_opts.add_argument('--testparam',
            help='comma separated list of parameters for tests (default: %(default)s)')
        new_opts.add_argument('--jdbc-path',
            help='path to the EXASOL JDBC Driver jar file (default: %(default)s)')
        new_opts.add_argument('--is-compat-mode',
            help='Compatibility mode (default: %(default)s)')
        new_opts.add_argument('--script-languages',
            help=' language definition, (default: %(default)s)')
        parser.set_defaults(
                lang=None,
                redirector_url=None,
                param=None,
                jdbc_path=None,
                is_compat_mode=None,
                script_languages=None,
                )

    def prepare_hook(self):
        global opts
        opts = self.opts
        if opts.lang is not None:
            client = ODBCClient(self.dsn, opts.user, opts.password)
            client.connect(autocommit=True)
            return load_functions(client=client, lang=opts.lang, schema='FN1', redirector=opts.redirector_url)
        return True

main = TestProgram

def load_functions(client, lang=None, schema='FN1', redirector=None):
    path = os.path.realpath(os.path.join(os.path.abspath(__file__),
            '../../../lang', lang))
    if not os.path.isdir(path):
        opts.log.critical('%s does not exits', path)
        return False
    _sql(client, 'DROP SCHEMA %s CASCADE' % schema, may_fail=True)
    _sql(client, 'CREATE SCHEMA %s' % schema, fatal_error=True)
    opts.log.info('searching for function definitions beneath %s', path)
    success = True
    for file in _walk(path):
        opts.log.info('loading functions from file %s', file)
        success = _load_file(client, file, redirector) and success
    client.commit()
    capabilities.extend([x[0]
            for x in client.query('''
                SELECT script_name
                    FROM EXA_USER_SCRIPTS
                    WHERE script_schema = ?''', schema)])
    client.close()
    return success

def _sql(client, sql, may_fail=False, fatal_error=False):
    try:
        opts.log.debug('executing SQL: %s', sql)
        client.query(sql)
        return True
    except pyodbc.Error as e:
        if may_fail:
            pass
        else:
            opts.log.critical(str(e))
            opts.log.exception(e)
            if not opts.log.isEnabledFor(logging.DEBUG):
                opts.log.error('SQL was: %s', sql)
            if fatal_error:
                sys.exit(1)
    return False

def _walk(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.sql'):
                yield os.path.join(root, f)

def _load_file(client, path, redirector=None):
    success = True
    for sql in _split_file_on_slash(path):
        sql = _rewrite_redirector(sql, redirector)
        success = _sql(client, sql) and success
    return success

def _rewrite_redirector(sql, redirector):
    if redirector is not None:
        return sql.replace('@@redirector_url@@',
                '\n# redirector '.join(redirector.split(',')))
    else:
        return sql

def _split_file_on_slash(path):
    sql = ''
    for line in open(path):
        line = line.decode('utf8')
        if line == '/\n':
            if sql:
                yield sql
            sql = ''
        else:
            sql += line
    if sql:
        yield sql

class TestCase(exatest.TestCase):
    def query(self, *args, **kwargs):
        new_args = list(args)
        new_args[0] = _rewrite_redirector(new_args[0], opts.redirector_url)
        return super(TestCase, self).query(*new_args, **kwargs)

    def query_via_exaplus(self, query):
        cmd = '''%(exaplus)s -c %(conn)s -u %(user)s -P %(password)s
                        -no-config -autocommit ON -L -pipe''' % {
                                'exaplus': os.environ.get('EXAPLUS'),
                                'conn': opts.server,
                                'user': self.user,
                                'password': self.password,
                                }
        print("Running the following exaplus command %s" % cmd)
        env = os.environ.copy()
        env['LC_ALL'] = 'en_US.UTF-8'
        exaplus = subprocess.Popen(
                    cmd.split(), 
                    env=env, 
                    stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
        langs=getScriptLanguagesFromArgs()
        query = "ALTER SESSION SET SCRIPT_LANGUAGES='%s';" % langs + "\n" + query
        print("Executing SQL in exaplus: %s" % query)
        out, err = exaplus.communicate(query.encode('utf8'))
        return out, err
    
    def import_via_exaplus(self, table_name, table_generator, prepare_sql):
        tmpdir = tempfile.mkdtemp()
        fifo_filename = os.path.join(tmpdir, 'myfifo')
        import_table_sql = '''IMPORT INTO %s FROM LOCAL CSV FILE '%s';'''%(table_name,fifo_filename)
        try:
            os.mkfifo(fifo_filename)
            write_trhead = threading.Thread(target=self._write_into_fifo, args=(fifo_filename, table_generator))
            write_trhead.start()
            sql=prepare_sql+"\n"+create_table_sql+"\n"+import_table_sql+"\n"+"commit;"
            out,err=self.query_via_exaplus(schema,sql)
            print(out)
            print(err)
            write_trhead.join()
        finally:
            os.remove(fifo_filename)
            os.rmdir(tmpdir)

    def _write_into_fifo(self, fifo_filename, table_generator):
        with open(fifo_filename,"w") as f:
            csvwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in table_generator:
                csvwriter.writerow(row)

    def import_via_insert(self, table_name, table_generator, column_names=None, tuples_per_insert=1000):
        if column_names is not None:
            column_names_str = ','.join(column_names)
        else:
            column_names_str = ''
        rows = []
        for i, row in enumerate(table_generator):
            values = ','.join(self._convert_insert_value(value) for value in row)
            row_str = "(%s)" % (values)
            rows.append(row_str)
            if i % tuples_per_insert == 0 and i > 0:
                self.run_insert(table_name,column_names_str,rows)
                del rows_str
                rows = []
        self.run_insert(table_name,column_names_str,rows)


    def run_insert(self, table_name, column_names_str, rows):
        if len(rows)>0:
            rows_str = ','.join(rows)
            if column_names_str != "":
                sql = "INSERT INTO %s (%s) VALUES %s" % (table_name, column_names_str, rows_str)
            else:
                sql = "INSERT INTO %s VALUES %s" % (table_name, rows_str)

            print("Executing insert statement %s"%sql)
            self.query(sql)

    def _convert_insert_value(self, value):
        if isinstance(value,str):
            return "'%s'" % value
        elif isinstance(value,(int,float)):
            return str(value)
        elif isinstance(value,bool):
            return "TRUE" if value else "FALSE"
        elif isinstance(value,Decimal):
            return str(value)
        elif isinstance(value,(date,datetime)):
            return "'%s'"%str(value)
        elif value is None:
            return "NULL";
        else:
            raise TypeError("Type %s of value %s is not supported" % (type(value), value))

# vim: ts=4:sts=4:sw=4:et:fdm=indent

