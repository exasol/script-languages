#!/usr/bin/env python2.7

import os
import getpass
import shutil
import subprocess
import sys
import tempfile
from textwrap import dedent
from contextlib import contextmanager

if os.path.exists('/usr/opt/testsystem/lib'):
    sys.path.insert(0, '/usr/opt/testsystem/lib')
else:
    sys.path.insert(0, '/opt/local/testsystem/lib')

try:
    import testsystem.unittest as unittest
except ImportError:
    import unittest

ROOTDIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../..'))


@contextmanager
def tempdir():
    prefix = getpass.getuser() + '.allinone_python_test-'
    tmp = tempfile.mkdtemp(prefix=prefix, dir='/var/tmp')
    try:
        yield tmp
    finally:
        pass #shutil.rmtree(tmp)


class AllInOneTest(unittest.TestCase):

    def allinone(self, script=None, spec='', indata='', outdata=''): 
        python = '/usr/bin/env python2.7'
        allinone = os.path.join(ROOTDIR,
                'tests/EngineTest/udf/bin/allinone_python.py')
        with tempdir() as tmp:
            cmd_name = os.path.join(tmp, 'allinone_python.py')
            if script is None:
                shutil.copy(allinone, cmd_name)
            else:
                with open(allinone) as instream, open(cmd_name, 'w') as outstream:
                    patchmode = False
                    for line in instream:
                        if line.startswith('#BEGIN'):
                            patchmode = True
                        if not patchmode:
                            outstream.write(line)
                        if line.startswith('#END'):
                            patchmode = False
                            outstream.write(spec)
                            outstream.write('\n#BEGIN\n')
                            outstream.write(script)
                            outstream.write('\n#END\n')
                    outstream.flush()
            outfile_name = os.path.join(tmp, 'outfile')
            infile_name = os.path.join(tmp, 'infile')
            with open(infile_name, 'w') as infile:
                infile.write(indata)
                infile.flush()
            child = subprocess.Popen([python, cmd_name, infile_name, outfile_name])
            child.wait()

            self.assertTrue(os.path.exists(outfile_name))
            content = open(outfile_name).read()
            self.assertEqual(outdata, content)

    def test_defaults(self):
        self.allinone(script=None, indata='1,2\n', outdata='"<1,2>"\r\n')

    
    def test_change_run_function(self):
        script = dedent('''\
            def run(ctx):
                return u"<%s,%s>" % (unicode(ctx.col2), unicode(ctx.col1))
            ''')
    
        self.allinone(script=script, indata='1,2\n', outdata='"<2,1>"\r\n')

    def test_different_spec(self):
        script = dedent('''\
            def run(ctx):
                return ctx.c2 * 100 + ctx.c1
            ''')
        spec = dedent('''\
            cfg_input_columns = [
                    {'name': 'c1',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    {'name': 'c2',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    ]
            cfg_output_columns = [
                    {'name': 'c1',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    ]
            cfg_input_column_count = len(cfg_input_columns)
            cfg_output_column_count = len(cfg_output_columns)
            ''')
        self.allinone(script=script, spec=spec,
                indata='1,2\n', outdata='201\r\n')

    def test_SCALAR_RETURNS(self):
        script = dedent('''\
            def run(ctx):
                return ctx.c1 + ctx.c2
            ''')
        spec = dedent('''\
            cfg_input_type = 'SCALAR'
            cfg_input_columns = [
                    {'name': 'c1',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    {'name': 'c2',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    ]
            cfg_output_type = 'RETURN'
            cfg_output_columns = [
                    {'name': 'c1',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    ]
            cfg_input_column_count = len(cfg_input_columns)
            cfg_output_column_count = len(cfg_output_columns)
            ''')
        self.allinone(script=script, spec=spec,
                indata='13,11\n', outdata='24\r\n')
            

    def test_SCALAR_EMITS(self):
        script = dedent('''\
            def run(ctx):
                ctx.emit(ctx.c2, ctx.c1)
            ''')
        spec = dedent('''\
            cfg_input_type = 'SCALAR'
            cfg_input_columns = [
                    {'name': 'c1',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    {'name': 'c2',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    ]
            cfg_output_type = 'EMITS'
            cfg_output_columns = [
                    {'name': 'c1',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    {'name': 'c2',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    ]
            cfg_input_column_count = len(cfg_input_columns)
            cfg_output_column_count = len(cfg_output_columns)
            ''')
        self.allinone(script=script, spec=spec,
                indata='1,2\n', outdata='2,1\r\n')
            
    def test_SET_RETURNS(self):
        script = dedent('''\
            def run(ctx):
                s = 0
                while True:
                    if ctx.c1 is not None:
                        s += ctx.c1 * ctx.c2
                    if not ctx.next():
                        break
                return s
            ''')
        spec = dedent('''\
            cfg_input_type = 'SET'
            cfg_input_columns = [
                    {'name': 'c1',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    {'name': 'c2',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    ]
            cfg_output_type = 'RETURN'
            cfg_output_columns = [
                    {'name': 'c1',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    ]
            cfg_input_column_count = len(cfg_input_columns)
            cfg_output_column_count = len(cfg_output_columns)
            ''')
        self.allinone(script=script, spec=spec,
                indata='2,3\n5,7\n', outdata='41\r\n')
            
    def test_SET_EMITS(self):
        script = dedent('''\
            def run(ctx):
                s = []
                while True:
                    if ctx.c1 is not None:
                        s.append([ctx.c1, ctx.c2])
                    if not ctx.next():
                        break
                ctx.emit(s[0][1], s[1][1])
                ctx.emit(s[0][0], s[1][0])
            ''')
        spec = dedent('''\
            cfg_input_type = 'SET'
            cfg_input_columns = [
                    {'name': 'c1',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    {'name': 'c2',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    ]
            cfg_output_type = 'EMITS'
            cfg_output_columns = [
                    {'name': 'c1',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    {'name': 'c2',
                     'type': int,
                     'sqltype': 'DECIMAL(18,0)',
                     'precision': 18,
                     'scale': 0,
                    },
                    ]
            cfg_input_column_count = len(cfg_input_columns)
            cfg_output_column_count = len(cfg_output_columns)
            ''')
        self.allinone(script=script, spec=spec,
                indata='2,3\n5,7\n', outdata='3,7\r\n2,5\r\n')

                                    

if __name__ == '__main__':
    if len(sys.argv) < 1:
        sys.exit("Usage: %s [rootdir [options]]" % sys.argv[0])
    if len(sys.argv) >= 1:
        ROOTDIR = sys.argv.pop(1)
    unittest.main()

# vim: ts=4:sw=4:sts=4:et:fdm=indent
