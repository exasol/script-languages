#!/usr/opt/bs-python-2.7/bin/python

import os
import sys
import socket
import base64
import quopri

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

from exatest.servers import MessageBox, HTTPServer, FTPServer, EchoServer, UDPEchoServer, SMTPServer
from exatest.utils import tempdir
from exatest.servers.ftpserver import DummyAuthorizer

class ExternalResourceTest(udf.TestCase):
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

class XMLProcessingTest(ExternalResourceTest):
    def test_xml_processing(self):
        '''DWA-13842'''
        self.query(udf.fixindent('''
                CREATE lua SCALAR SCRIPT
                process_users(url VARCHAR(300))
                EMITS (firstname VARCHAR(100), lastname VARCHAR(100)) AS

                require("lxp")
                http = require("socket.http")
            
                local in_user_tag = false;                
                local in_first_name_tag = false;
                local in_family_name_tag = false;
                local current = {}
                local users = {}
                p = lxp.new({StartElement = function(p,tag,attr)
                                                if tag == "user" and attr.active == "1" then in_user_tag = true; end
                                                if tag == "first_name" then in_first_name_tag = true; end
                                                if tag == "family_name" then in_family_name_tag = true; end 
                                            end,
                             EndElement = function(p, tag)
                                                if tag == "user" then in_user_tag = false; end
                                                if tag == "first_name" then in_first_name_tag = false; end
                                                if tag == "family_name" then in_family_name_tag = false; end
                                          end,
                             CharacterData = function(p, txt)
                                                if in_user_tag then
                                                    if in_first_name_tag then current.first_name = txt end
                                                    if in_family_name_tag then current.family_name = txt end
                                                end
                                                if current.first_name and current.family_name then
                                                   users[#users+1] = current
                                                   current = {}
                                                end
                                             end})
                function run(ctx)
                    data = http.request(ctx.url)
                    p:parse(data); p:parse(); p:close();
                    for i=1,#users do
                        ctx.emit(users[i].first_name, users[i].family_name)
                    end
                end
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


class FTPServerTest(ExternalResourceTest):
    def test_xml_processing(self):
        '''DWA-13842'''
        self.query(udf.fixindent('''
                CREATE lua SCALAR SCRIPT
                process_users(url VARCHAR(200))
                EMITS (firstname VARCHAR(10000), lastname VARCHAR(10000)) AS
                      
                require("lxp")
                ftp = require("socket.ftp")
                local in_user_tag = false;                
                local in_first_name_tag = false;
                local in_family_name_tag = false;
                local current = {}
                local users = {}
                p = lxp.new({StartElement = function(p,tag,attr)
                                                if tag == "user" and attr.active == "1" then in_user_tag = true; end
                                                if tag == "first_name" then in_first_name_tag = true; end
                                                if tag == "family_name" then in_family_name_tag = true; end 
                                            end,
                             EndElement = function(p, tag)
                                                if tag == "user" then in_user_tag = false; end
                                                if tag == "first_name" then in_first_name_tag = false; end
                                                if tag == "family_name" then in_family_name_tag = false; end
                                          end,
                             CharacterData = function(p, txt)
                                                if in_user_tag then
                                                    if in_first_name_tag then current.first_name = txt end
                                                    if in_family_name_tag then current.family_name = txt end
                                                end
                                                if current.first_name and current.family_name then
                                                   users[#users+1] = current
                                                   current = {}
                                                end
                                             end})
 
                function run(ctx)
                    data = ftp.get(ctx.url)
                    p:parse(data); p:parse(); p:close();
                    for i=1,#users do
                        ctx.emit(users[i].first_name, users[i].family_name)
                    end
                end
                '''))
            
        with tempdir() as tmp:
            with open(os.path.join(tmp, 'keepers.xml'), 'w') as f:
                f.write(self.xml())
            
            with FTPServer(tmp) as ftpd:
                url = 'ftp://anonymous:guest@%s:%d/keepers.xml' % ftpd.address
                rows = self.query('''
                        SELECT process_users('%s')
                        FROM DUAL
                        ORDER BY lastname
                        ''' % url)
            
        expected = [('Joe', 'Hart'), ('Manuel', 'Neuer')]
        self.assertRowsEqual(expected, rows)

    def test_put_method(self):
        self.query(udf.fixindent('''
            create lua scalar script
            ftp_put(url varchar(200), text varchar(2000))
            returns double
            as
            ftp = require("socket.ftp")
            function run(ctx)
                local ret = ftp.put(ctx.url, ctx.text)
                return ret
            end
            /
            '''))

        with tempdir() as tmp:
            auth = DummyAuthorizer()
            auth.add_user('user', 'passwd', tmp, perm='elradfmw')
            with FTPServer(tmp, authorizer=auth) as ftpd:
                url = 'ftp://user:passwd@%s:%d/WRITTENBYLUA' % ftpd.address
                self.query('''
                        SELECT ftp_put('%s', 'some text written by a lua script via ftp')
                        FROM DUAL
                        ''' % url)

            with open(os.path.join(tmp, 'WRITTENBYLUA'), 'r') as f:
                read_data = f.read()

            self.assertEqual('some text written by a lua script via ftp', read_data)


class CleanupTest(udf.TestCase):
    maxDiff = None

    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')
    
    def test_cleanup_is_called_at_least_once(self):
        with MessageBox() as mb:
            self.query(udf.fixindent('''
                CREATE lua SCALAR SCRIPT
                sendmail(host VARCHAR(200), port VARCHAR(5), msg VARCHAR(200))
                RETURNS DOUBLE AS

                require("socket")
 
                function run(ctx)
                    host = ctx.host
                    port = ctx.port
                    msg = ctx.msg
                    return 0
                end

                function cleanup()
                    if host then
                        sock = assert(socket.tcp())
                        assert(sock:connect(host, port))
                        sock:send(msg)
                        sock:close()
                    end
                end
                '''))
            self.query('''SELECT sendmail('%s', '%d', 'foobar') FROM DUAL''' %
                    mb.address)

        self.assertIn('foobar', mb.data)

    def test_cleanup_is_called_exactly_once_for_each_vm(self):
        with MessageBox() as mb:

            host, port = mb.address
            self.query(udf.fixindent('''
                CREATE lua SCALAR SCRIPT
                sendmail(dummy DOUBLE)
                RETURNS DOUBLE AS

                require('socket')
        
                msg = math.random()

                sock = assert(socket.tcp())
                assert(sock:connect('%s', %d))
                sock:send('init:' .. msg)
                sock:close()

                function run(ctx)
                    return 0
                end

                function cleanup()
                    sock = assert(socket.tcp())
                    assert(sock:connect('%s', %d))
                    sock:send('cleanup:' .. msg)
                    sock:close()
                end
                ''' % (host, port, host, port)))
            self.query('''
                SELECT max(sendmail(float1))
                FROM test.enginetablebig1''') 

        data = mb.data
        self.assertGreater(len(data), 0)
        init = sorted([x.split(':')[1] for x in data if x.startswith('init')])
        cleanup = sorted([x.split(':')[1] for x in data if x.startswith('cleanup')])
        self.assertEquals(init, cleanup)
        # FIXME: math.random() is not thread-unique
        #self.assertEquals(sorted(set(init)), init)


class SocketSendRecvTest(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')

    def test_socket_send_recv(self):
        with EchoServer() as echos:
            host, port = echos.address
            sentdata = 'lets send this to an echo server'

            self.query(udf.fixindent('''
                create lua scalar script
                send_recv(indata varchar(4096))
                returns varchar(4096)
                as
                socket = require("socket")
                function run(ctx)
                    sock = assert(socket.tcp())
                    assert(sock:connect('%s', %d))
                    sock:send(ctx.indata)
                    out = sock:receive('*a')
                    sock:close()
                    return out
                end
                /
                ''' % (host, port)))
            rows = self.query('''select send_recv('%s') res from dual;''' % (sentdata))
            self.assertEqual(1, self.rowcount())
            self.assertEquals(rows[0].RES, sentdata)


class UDPSocketTest(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')

    def test_udp_socket_send_recv(self):
        with UDPEchoServer() as echos:
            host, port = echos.address
            sentdata = 'lets send this to an UDP echo server'

            self.query(udf.fixindent('''
                create lua scalar script
                udp_socket(text varchar(1024))
                returns varchar(1024)
                as
                socket = require("socket")
                function run(ctx)
                        udp = assert(socket.udp())
                        assert(udp:sendto(ctx.text,'%s', %d))
                        local r = assert(udp:receive())
                        return r
                end
                /
                ''' % (host, port)))
            rows = self.query('''select udp_socket('%s') res from dual;''' % (sentdata))
            self.assertEqual(1, self.rowcount())
            self.assertEquals(rows[0].RES, sentdata)


class DNSTest(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')

    def test_dns_functions(self):
        self.query(udf.fixindent('''
            create lua scalar script
            dns()
            emits (r varchar(1000))
            as
            socket = require("socket")
            function run(ctx)
                hostname = socket.dns.gethostname()
                ctx.emit(hostname)
                ip = socket.dns.toip(hostname)
                ctx.emit(ip)
                reshostname = socket.dns.tohostname(ip)
                ctx.emit(reshostname)
            end
            /
            '''))
        rows = self.query('''select dns() from dual;''')

        self.assertEqual(3, self.rowcount())
        self.assertEqual(socket.gethostbyname(rows[0].R), rows[1].R)
        self.assertEqual(socket.gethostbyaddr(socket.gethostbyname(rows[0].R))[0], rows[2].R)


class MIMETest(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')

    def test_base64_encode(self):
        self.query(udf.fixindent('''
            create lua scalar script
            b64enc(text varchar(1024))
            returns varchar(1024)
            as
            function run(ctx)
                    mime = require("mime")
                    return (mime.b64(ctx.text))
            end
            /
            '''))
        rows = self.query('''select b64enc('a,b,c as easy as 1,2,3') r from dual;''')
        self.assertEqual(1, self.rowcount())
        self.assertEqual( base64.b64encode('a,b,c as easy as 1,2,3'), rows[0].R )

    def test_base64_decode(self):
        self.query(udf.fixindent('''
            create lua scalar script
            b64dec(text varchar(1024))
            returns varchar(1024)
            as
            function run(ctx)
                    mime = require("mime")
                    return (mime.unb64(ctx.text))
            end
            /
            '''))
        rows = self.query('''select b64dec('YSxiLGMgYXMgZWFzeSBhcyAxLDIsMw==') r from dual;''')
        self.assertEqual(1, self.rowcount())
        self.assertEqual( base64.b64decode('YSxiLGMgYXMgZWFzeSBhcyAxLDIsMw=='), rows[0].R )

    def test_quotprint_encode(self):
        self.query(udf.fixindent('''
            create lua scalar script
            qpenc(text varchar(1024))
            returns varchar(1024)
            as
            function run(ctx)
                    mime = require("mime")
                    return (mime.qp(ctx.text))
            end
            /
            '''))
        rows = self.query('''select qpenc('ein = und noch eines=') r from dual;''')
        self.assertEqual(1, self.rowcount())
        self.assertEqual( quopri.encodestring('ein = und noch eines='), rows[0].R )

    def test_quotprint_decode(self):
        self.query(udf.fixindent('''
            create lua scalar script
            qpdec(text varchar(1024))
            returns varchar(1024)
            as
            function run(ctx)
                    mime = require("mime")
                    return (mime.unqp(ctx.text))
            end
            /
            '''))
        rows = self.query('''select qpdec('ein =3D und noch eines=3D') r from dual;''')
        self.assertEqual(1, self.rowcount())
        self.assertEqual( quopri.decodestring('ein =3D und noch eines=3D'), rows[0].R )

class SMTPTest(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')

    def test_smtp_functions(self):
        self.query(udf.fixindent('''
            create lua scalar script
            send_mail(host varchar(100), port double)
            returns varchar(1000)
            as
            smtp = require("socket.smtp")
            function run(ctx)
                from = "<sender@ex.com>"
                to = { "<recipient@ex.com>" }
                mesgt = {
                    headers = {
                        to = "Mr. X <recipient@ex.com>",
                        subject = "a message from EXASolution"
                    }, 
                        body = "this is a body text"
                }
            r, e = smtp.send{
                from = from,
                rcpt = to,
                source = smtp.message(mesgt),
                server = ctx.host,
                port = ctx.port
            }
            if not r then 
                return e
            else
                return "success"
            end
            end
            /
            '''))

        #with SMTPServer(debug=True) as smtpd:
        with SMTPServer() as smtpd:
            host, port = smtpd.address
            query_string = "select send_mail('" + host + "'," + str(port) + ") from dual;"
            rows = self.query(query_string)

        self.assertEqual(1, len(smtpd.messages))
        self.assertEqual('sender@ex.com', smtpd.messages[0].sender)
        self.assertIn('recipient@ex.com', smtpd.messages[0].recipients)
        self.assertIn('content-type: text/plain; charset="iso-8859-1"', smtpd.messages[0].body)
        self.assertIn('x-mailer: LuaSocket 2.0.2', smtpd.messages[0].body)
        self.assertIn('subject: a message from EXASolution', smtpd.messages[0].body)
        self.assertIn('date:', smtpd.messages[0].body)
        self.assertIn('this is a body text', smtpd.messages[0].body)

        self.assertEqual(1, self.rowcount())

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

