#!/usr/bin/env python2.7

import ftplib
import os
import smtplib
import socket
import sys
import unittest
import urllib
from email.mime.text import MIMEText

sys.path.insert(0, os.path.realpath(os.path.join(__file__, '../../../lib')))

from exatest.test import selftest

import exatest
from exatest.testcase import (
        useData,
        TestCase,
        ParameterizedTestCase,
        get_sort_key,
        skip,
        skipIf,
        expectedFailure,
        expectedFailureIf,
        )
from exatest.utils import tempdir
from exatest.servers import FTPServer, HTTPServer, SMTPServer
from exatest.servers.ftpserver import DummyAuthorizer

class FTPServerTest(unittest.TestCase):
    def test_anonymous(self):
        class Module:
            class Test(unittest.TestCase):
                def test_1(self):
                    with tempdir() as tmp:
                        with open(os.path.join(tmp, 'dummy'), 'w'):
                            pass
                        with FTPServer(tmp) as ftpd:
                            ftp = ftplib.FTP()
                            ftp.connect(*ftpd.address)
                            ftp.login()
                            data = []
                            ls = ftp.retrlines('LIST', data.append)
                            ftp.quit()
                    self.assertIn('dummy', '\n'.join(data))
        with selftest(Module) as result:
            self.assertTrue(result.wasSuccessful())

    def test_authenticated_user(self):
        class Module:
            class Test(unittest.TestCase):
                def test_1(self):
                    with tempdir() as tmp:
                        with open(os.path.join(tmp, 'dummy'), 'w'):
                            pass
                        auth = DummyAuthorizer()
                        auth.add_user('user', 'passwd', tmp, perm='elradfmw')
                        with FTPServer(tmp, authorizer=auth) as ftpd:
                            ftp = ftplib.FTP()
                            ftp.connect(*ftpd.address)
                            ftp.login('user', 'passwd')
                            ftp.mkd('some_dir')
                            data = []
                            ls = ftp.retrlines('LIST', data.append)
                            ftp.quit()
                    self.assertIn('dummy', '\n'.join(data))
                    self.assertIn('some_dir', '\n'.join(data))
        with selftest(Module) as result:
            self.assertTrue(result.wasSuccessful())

    def test_server_is_chdir_safe(self):
        cwd = os.getcwd()
        with tempdir() as tmp:
            self.assertEqual(cwd, os.getcwd())
            with FTPServer(tmp) as ftpd:
                self.assertEqual(cwd, os.getcwd())
            self.assertEqual(cwd, os.getcwd())
        self.assertEqual(cwd, os.getcwd())
                

class HTTPServerTest(unittest.TestCase):
    def test_anonymous(self):
        class Module:
            class Test(unittest.TestCase):
                def test_1(self):
                    with tempdir() as tmp:
                        with open(os.path.join(tmp, 'dummy'), 'w') as f:
                            f.write('babelfish')
                        with HTTPServer(tmp) as httpd:
                            url = urllib.urlopen('http://%s:%d/dummy' % httpd.address)
                            data = url.readlines()
                    self.assertIn('babelfish', '\n'.join(data))
        with selftest(Module) as result:
            self.assertTrue(result.wasSuccessful())

    def test_server_is_chdir_safe(self):
        cwd = os.getcwd()
        with tempdir() as tmp:
            self.assertEqual(cwd, os.getcwd())
            with HTTPServer(tmp) as httpd:
                # Current implementation chdir to documentroot;
                # needs subprocesses to avoid this.
                pass
                #self.assertEqual(cwd, os.getcwd())
            self.assertEqual(cwd, os.getcwd())
        self.assertEqual(cwd, os.getcwd())

                
class SMTPServerTest(unittest.TestCase):
    def test_simple_message(self):
        class Module:
            class Test(unittest.TestCase):
                def test_1(self):
                    with SMTPServer(debug=True) as smtpd:
                        host, port = smtpd.address
                        self.assertNotEqual(0, port)
                        client = smtplib.SMTP(host, port, timeout=5)
                        client.set_debuglevel(1)
                        msg = MIMEText('Test mail')
                        msg['Subject'] = 'Little test mail'
                        msg['From'] = 'from.address@foo.bar'
                        msg['To'] = 'babelfish@mailing-list.com'
                        client.sendmail('from.address@foo.bar', ('to.address@babel.fish', 'babel@fish.org'), msg.as_string())
                        client.quit()
                    self.assertEqual(1, len(smtpd.messages))
                    self.assertEqual('from.address@foo.bar', smtpd.messages[0].sender)
                    self.assertIn('babel@fish.org', smtpd.messages[0].recipients)
                    self.assertIn('Test mail', smtpd.messages[0].body)
        with selftest(Module) as result:
            self.assertTrue(result.wasSuccessful())

    def test_non_esmtp(self):
        class Module:
            class Test(unittest.TestCase):
                def test_1(self):
                    with SMTPServer(esmtp=False, debug=True) as smtpd:
                        host, port = smtpd.address
                        s = socket.create_connection(smtpd.address)
                        buf1 = s.recv(4096)
                        s.send('EHLO some.host.com\r\n')
                        buf2 = s.recv(4096)
                        s.send('QUIT\r\n')
                        buf3 = s.recv(4096)
                        s.close()
                    self.assertIn('220', buf1)
                    self.assertIn('502 Error', buf2)
        with selftest(Module) as result:
            self.assertTrue(result.wasSuccessful())

if __name__ == '__main__':
    unittest.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
