from __future__ import absolute_import

import SimpleHTTPServer
import SocketServer
import asyncore
import collections
import os
import select
import smtpd
import socket
import sys
import threading

from ..threading import Thread
from .. import utils

class BaseSimpleServer(object):
    def __init__(self):
        self._thread = None
        self.host = None
        self.port = None

    @property
    def address(self):
        return (self.host, self.port)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        pass

    def stop(self):
        self._thread.shutdown()
        self._thread.join(5)

class MessageBox(BaseSimpleServer):
    def _messagebox(self, s):
        slf = threading.current_thread()
        slf.data = []
        s.listen(20)
        while not slf.shutdown_requested():
            if s in select.select([s], [], [], 1)[0]:
                sock, addr = s.accept()
                slf.data.append(sock.recv(4096))
                sock.close()
        s.close()
    
    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        self._thread = Thread(target=self._messagebox, args=(s,))
        self._thread.start()
        self.host = socket.gethostbyname(socket.getfqdn())
        self.port = s.getsockname()[1]
        print('host: {}, sn.host: {}, sn.port: {}'.format(self.host, s.getsockname()[0], self.port))

    @property
    def data(self):
        return self._thread.data

class EchoServer(BaseSimpleServer):
    def _echo(self, s):
        slf = threading.current_thread()
        s.listen(20)
        while not slf.shutdown_requested():
            if s in select.select([s], [], [], 1)[0]:
                sock, addr = s.accept()
                data = sock.recv(4096)
                if (data):
                    sock.send(data)
                sock.close()
        s.close()

    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        self._thread = Thread(target=self._echo, args=(s,))
        self._thread.start()
        self.host = socket.gethostbyname(socket.getfqdn())
        self.port = s.getsockname()[1]

class UDPEchoServer(BaseSimpleServer):
    def _echo(self, s):
        slf = threading.current_thread()
        while not slf.shutdown_requested():
            if s in select.select([s], [], [], 1)[0]:
                data, addr = s.recvfrom(1024)
                s.sendto(data,addr)
        s.close()

    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('', 0))
        self._thread = Thread(target=self._echo, args=(s,))
        self._thread.start()
        self.host = socket.gethostbyname(socket.getfqdn())
        self.port = s.getsockname()[1]

class _ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass

class HTTPServer(BaseSimpleServer):
    def __init__(self, documentroot='.'):
        super(HTTPServer, self).__init__()
        self.documentroot = documentroot
        self._server = None

    def _httpserver(self):
        with utils.chdir(self.documentroot):
            self._server.serve_forever()        

    def start(self):
        handler = SimpleHTTPServer.SimpleHTTPRequestHandler
        self._server = _ThreadedTCPServer(('', 0), handler)
        
        self._thread = Thread(target=self._httpserver, args=())
        self._thread.start()
        self.host = socket.gethostbyname(socket.getfqdn())
        self.port = self._server.server_address[1]

    def stop(self):
        self._server.shutdown()
        super(HTTPServer, self).stop()

class FTPServer(BaseSimpleServer):
    def __init__(self, documentroot='.', authorizer=None):
        global ftpserver
        from . import servers
        from . import authorizers
        super(FTPServer, self).__init__()
        self.documentroot = documentroot
        if authorizer is None:
            self.authorizer = authorizers.DummyAuthorizer()
            self.authorizer.add_anonymous(self.documentroot)
        else:
            self.authorizer = authorizer
    
    def _ftpserver(self):
        self._server.serve_forever()

    def start(self):
        from . import handlers
        ftp_handler = handlers.FTPHandler
        ftp_handler.authorizer = self.authorizer
        self._server = servers.FTPServer(('', 0), ftp_handler)
        
        self._thread = Thread(target=self._ftpserver, args=())
        self._thread.start()
        self.host = socket.gethostbyname(socket.getfqdn())
        self.port = self._server.address[1]

    def stop(self):
        self._server.close_all()
        super(FTPServer, self).stop()


class _SMTPChannel(smtpd.SMTPChannel):
    '''EHLO is missing in smtpd.SMTPChannel'''
    def smtp_EHLO(self, arg):
        if not arg:
            self.push('501 Syntax: EHLO hostname')
            return
        if self.__greeting:
            self.push('503 Duplicate HELO/EHLO')
        else:
            self.__greeting = arg
            self.push('250 %s' % self.__fqdn)

class _SMTPServer(smtpd.SMTPServer):
    '''Use private map; ignore remoteaddr'''
    def __init__(self, localaddr, map, debug=False, esmtp=True):
        self._localaddr = localaddr
        self._remoteaddr = None
        self.__smtpchannel = _SMTPChannel if esmtp else smtpd.SMTPChannel
        self.__debug = debug
        self.__messages = []
        asyncore.dispatcher.__init__(self, map=map)
        try:
            self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
            # try to re-use a server port if possible
            self.set_reuse_addr()
            self.bind(localaddr)
            self.listen(5)
        except:
            # cleanup asyncore.socket_map before raising
            self.close()
            raise

    def handle_accept(self):
        pair = self.accept()
        if pair is not None:
            conn, addr = pair
            channel = self.__smtpchannel(self, conn, addr)

    _message = collections.namedtuple('Message', 'host, sender, recipients, body')

    def process_message(self, peer, mailfrom, rcpttos, data):
        self.__messages.append(self._message(peer, mailfrom, rcpttos, data))
        if self.__debug:
            print '---------- MESSAGE ENVELOPE ---------'
            print 'Peer:', peer
            print 'From:', mailfrom
            print 'To:', ', '.join(rcpttos)
            inheaders = 1
            lines = data.split('\n')
            print '---------- MESSAGE FOLLOWS ----------'
            for line in lines:
                # headers first
                if inheaders and not line:
                    print 'X-Peer:', peer[0]
                    inheaders = 0
                print line
            print '------------ END MESSAGE ------------'



class SMTPServer(BaseSimpleServer):
    def __init__(self, debug=False, esmtp=True):
        super(SMTPServer, self).__init__()
        self.esmtp = esmtp
        self.debug = debug
        self._server = None
        self._wait_for_server = threading.Semaphore(0)

    def _smtpserver(self):
        map = None
        self._server = _SMTPServer(('', 0), map, debug=self.debug, esmtp=self.esmtp)
        self._wait_for_server.release()
        asyncore.loop(timeout=1, map=map)

    def start(self):
        self._thread = Thread(target=self._smtpserver, args=())
        self._thread.start()
        self.host = socket.gethostbyname(socket.getfqdn())
        with self._wait_for_server:
            pass
        self.port = self._server.socket.getsockname()[1]

    def stop(self):
        self._server.close()
        super(SMTPServer, self).stop()

    @property
    def messages(self):
        return self._server.__messages

# vim: ts=4:sts=4:sw=4:et:fdm=indent
__ver__ = '1.5.4'
__author__ = "Giampaolo Rodola' <g.rodola@gmail.com>"
__web__ = 'https://github.com/giampaolo/pyftpdlib/'
