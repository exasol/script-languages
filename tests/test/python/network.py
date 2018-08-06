#!/usr/bin/env python2.7

import os
import sys
import urllib

import lxml.etree as etree

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

from exatest.servers import HTTPServer, MessageBox
from exatest.utils import tempdir

class HTTPTest(udf.TestCase):
    def test_selftest(self):
        with tempdir() as tmp:
            with open(os.path.join(tmp, 'foo.xml'), 'w') as f:
                f.write('''<foo/>\n''')
            with HTTPServer(tmp) as hs:
                self.assertIn('<foo/>',
                    urllib.urlopen('http://%s:%d/foo.xml' % hs.address).read())

class XMLProcessingTest(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')
   
    def xml(self):
        return udf.fixindent('''\
                <?xml version='1.0' encoding='UTF-8'?>
                <users>
                    <user active="1">
                        <first_name>Manuel</first_name>
                        <family_name>Neuer</family_name>
                    </user>
                    <user active="1">
                        <first_name>Joe</first_name>
                        <family_name>Hart</family_name>
                    </user>
                    <user active="0">
                        <first_name>Oliver</first_name>
                        <family_name>Kahn</family_name>
                    </user>
                </users>
                ''')

    def xmlns(self):
        return udf.fixindent('''\
                <?xml version='1.0' encoding='UTF-8'?>
                <users xmlns:foo="http://foo/bar" xmlns="http://default/">
                    <user active="1">
                        <first_name>Manuel</first_name>
                        <foo:family_name>Neuer</foo:family_name>
                    </user>
                    <user active="1">
                        <first_name>Joe</first_name>
                        <foo:family_name>Hart</foo:family_name>
                    </user>
                    <user active="0">
                        <first_name>Oliver</first_name>
                        <foo:family_name>Kahn</foo:family_name>
                    </user>
                </users>
                ''')

    def test_dry_run(self):
        tree = etree.XML(self.xml())
        result = []
        for u in tree.findall('user/[@active="1"]'):
            result.append((u.findtext('first_name'), u.findtext('family_name')))
        expected = [('Joe', 'Hart'), ('Manuel', 'Neuer')]
        self.assertEqual(expected, sorted(result))
        


    def test_xml_processing(self):
        self.query(udf.fixindent('''
                CREATE python SCALAR SCRIPT
                process_users(url VARCHAR(200))
                EMITS (firstname VARCHAR(100), lastname VARCHAR(100)) AS

                import urllib
                import lxml.etree as etree
                # import xml.etree.cElementTree as etree


                def run(ctx):
                    data = ''.join(urllib.urlopen(ctx.url).readlines())
                    tree = etree.XML(data)
                    for user in tree.findall('user/[@active="1"]'):
                        fn = user.findtext('first_name')
                        ln = user.findtext('family_name')
                        ctx.emit(fn, ln)
                '''))
            
        with tempdir() as tmp:
            with open(os.path.join(tmp, 'keepers.xml'), 'w') as f:
                f.write(self.xml())
            
            with HTTPServer(tmp) as hs:
                url = 'http://%s:%d/keepers.xml' % hs.address
                rows = self.query('''
                        SELECT process_users('%s')
                        FROM DUAL
                        ORDER BY lastname
                        ''' % url)
            
        expected = [('Joe', 'Hart'), ('Manuel', 'Neuer')]
        self.assertRowsEqual(expected, rows)

    def test_xmlns_processing(self):
        self.query(udf.fixindent('''
                CREATE python SCALAR SCRIPT
                process_users(url VARCHAR(200))
                EMITS (firstname VARCHAR(100), lastname VARCHAR(100)) AS

                import urllib
                import lxml.etree as etree
                # import xml.etree.cElementTree as etree
       
                def run(ctx):
                    data = ''.join(urllib.urlopen(ctx.url).readlines())
                    tree = etree.XML(data)
                    for user in tree.findall('{http://default/}user/[@active="1"]'):
                        fn = user.findtext('{http://default/}first_name')
                        ln = user.findtext('{http://foo/bar}family_name')
                        ctx.emit(fn, ln)
                '''))
            
        with tempdir() as tmp:
            with open(os.path.join(tmp, 'keepers.xml'), 'w') as f:
                f.write(self.xmlns())
            
            with HTTPServer(tmp) as hs:
                url = 'http://%s:%d/keepers.xml' % hs.address
                rows = self.query('''
                        SELECT process_users('%s')
                        FROM DUAL
                        ORDER BY lastname
                        ''' % url)
            
        expected = [('Joe', 'Hart'), ('Manuel', 'Neuer')]
        self.assertRowsEqual(expected, rows)

class CleanupTest(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')
    
    def test_cleanup_is_called_at_least_once(self):
        with MessageBox() as mb:
            host, port = mb.address

            self.query(udf.fixindent('''
                CREATE python SCALAR SCRIPT
                sendmail(host VARCHAR(200), port INT, msg VARCHAR(200))
                RETURNS INT AS

                import socket
        
                host = None
                port = None
                msg = None

                def run(ctx):
                    global host, port, msg
                    host = ctx.host
                    port = ctx.port
                    msg = ctx.msg
                    return 0

                def cleanup():
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((host, port))
                    sock.send(msg)
                    sock.close()
                '''))
            self.query('''SELECT sendmail('%s', %d, 'foobar') FROM DUAL''' %
                    (host, port))

        self.assertIn('foobar', mb.data)

    def test_cleanup_is_called_exactly_once_for_each_vm(self):
        with MessageBox() as mb:
            host, port = mb.address

            print('test: host {}, port {}'.format(host, port))

            self.query(udf.fixindent('''
                CREATE python SCALAR SCRIPT
                sendmail(dummy DOUBLE)
                RETURNS INT AS

                import socket
                import uuid
        
                msg = str(uuid.uuid4())

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(('%s', %d))
                sock.send('init:' + msg)
                sock.close()

                def run(ctx):
                    return 0

                def cleanup():
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect(('%s', %d))
                    sock.send('cleanup:' + msg)
                    sock.close()
                ''' % (host, port, host, port)))
            self.query('''
                SELECT max(sendmail(float1))
                FROM test.enginetablebig1''') 

        data = mb.data
        self.assertGreater(len(data), 0)
        init = sorted([x.split(':')[1] for x in data if x.startswith('init')])
        cleanup = sorted([x.split(':')[1] for x in data if x.startswith('cleanup')])
        self.assertEquals(init, cleanup)
        self.assertEquals(sorted(set(init)), init)

    def test_cleanup_is_called_exactly_once_for_each_vm_with_crash_in_run(self):
        with MessageBox() as mb:
            host, port = mb.address

            self.query(udf.fixindent('''
                CREATE python SCALAR SCRIPT
                sendmail(dummy DOUBLE)
                RETURNS INT AS

                import socket
                import uuid
        
                msg = str(uuid.uuid4())

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(('%s', %d))
                sock.send('init:' + msg)
                sock.close()

                def run(ctx):
                    raise ValueError(42)

                def cleanup():
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect(('%s', %d))
                    sock.send('cleanup:' + msg)
                    sock.close()
                ''' % (host, port, host, port)))
            with self.assertRaises(Exception):
                self.query('''
                    SELECT max(sendmail(float1))
                    FROM test.enginetablebig1''') 

        data = mb.data
        self.assertGreater(len(data), 0)
        init = sorted([x.split(':')[1] for x in data if x.startswith('init')])
        cleanup = sorted([x.split(':')[1] for x in data if x.startswith('cleanup')])
        self.assertEquals(init, cleanup)
        self.assertEquals(sorted(set(init)), init)

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

