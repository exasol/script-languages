#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

from exatest.servers import MessageBox, HTTPServer, FTPServer
from exatest.utils import tempdir

class ExternalResourceTest(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA t0 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t0')
   
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

#for (i in 1:length(tree$doc$children$users)) {if (tree$doc$children$users[i]$user$attributes['active']==1) {firstname <- tree$doc$children$users[i]$user$children$first_name$children$text; familyname <- tree$doc$children$users[i]$user$children$family_name$children$text; print(firstname); print(familyname);}}

class XMLProcessingTest(ExternalResourceTest):
    def test_xml_processing(self):
        '''DWA-13842'''
        self.query(udf.fixindent('''
                CREATE or REPLACE R SCALAR SCRIPT
                process_users(url VARCHAR(300))
                EMITS (firstname VARCHAR(100), lastname VARCHAR(100)) as
                require('RCurl')
                require('XML')
                run <- function(ctx) {
                    cont <- getURL(ctx$url)
                    tree <- xmlTreeParse(cont)
                    for (i in 1:length(tree$doc$children$users)) {
                        if (tree$doc$children$users[i]$user$attributes['active']==1) {
                                firstname <- tree$doc$children$users[i]$user$children$first_name$children$text$value;
                                familyname <- tree$doc$children$users[i]$user$children$family_name$children$text$value;
                                ctx$emit(firstname, familyname)
                        }
                    }
                }
                '''))
            
        with tempdir() as tmp:
            with open(os.path.join(tmp, 'keepers.xml'), 'w') as f:
                f.write(self.xml())
            
            with HTTPServer(tmp) as hs:
                url = 'http://%s:%d/keepers.xml' % hs.address
                query = '''
                        SELECT process_users('%s')
                        FROM DUAL
                        ORDER BY lastname
                        ''' % url
                rows = self.query(query)
                expected = [('Joe', 'Hart'), ('Manuel', 'Neuer')]
                self.assertRowsEqual(expected, rows)


class FTPServerTest(ExternalResourceTest):
    def test_xml_processing(self):
        '''DWA-13842'''
        self.query(udf.fixindent('''
                CREATE OR REPLACE R SCALAR SCRIPT
                process_users(url VARCHAR(200))
                EMITS (firstname VARCHAR(100), lastname VARCHAR(100)) AS
 
                require('RCurl')
                require('XML')
                run <- function(ctx) {
                    cont <- getURL(ctx$url)
                    tree <- xmlTreeParse(cont)
                    for (i in 1:length(tree$doc$children$users)) {
                        if (tree$doc$children$users[i]$user$attributes['active']==1) {
                                firstname <- tree$doc$children$users[i]$user$children$first_name$children$text$value;
                                familyname <- tree$doc$children$users[i]$user$children$family_name$children$text$value;
                                ctx$emit(firstname, familyname)
                        }
                    }
                }
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

class CleanupTest(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')
    
    def test_cleanup_is_called_at_least_once(self):
        with MessageBox() as mb:
            self.query(udf.fixindent('''
               CREATE R SCALAR SCRIPT
                sendmail(host VARCHAR(200), port VARCHAR(5), msg VARCHAR(200))
                RETURNS DOUBLE AS

                my_host <- 0
                my_port <- 0
                my_msg <- 0
 
                run <- function(ctx) {
                    my_host <<- ctx$host
                    my_port <<- ctx$port
                    my_msg <<- ctx$msg
                    0
                }

                cleanup <- function() {
                   sock <- make.socket(my_host, as.integer(my_port))
                   write.socket(sock, my_msg)
                   close.socket(sock)
                }
                '''))
            self.query('''SELECT sendmail('%s', '%d', 'foobar') FROM DUAL''' %
                    mb.address)

        self.assertIn('foobar', mb.data)


    def test_cleanup_is_called_exactly_once_for_each_vm(self):
        with MessageBox() as mb:
            host, port = mb.address
            self.query(udf.fixindent('''
                CREATE R SCALAR SCRIPT
                sendmail(dummy DOUBLE)
                RETURNS DOUBLE AS
        
                msg <- runif(1)

                sock <- make.socket('%s', %d)
                write.socket(sock, paste('init:',  msg, sep = ''))
                close.socket(sock)

                run <- function(ctx) {
                    0
                }

                cleanup <- function() {
                    sock <- make.socket('%s', %d)
                    write.socket(sock, paste('cleanup:', msg, sep=''))
                    close.socket(sock)
                }
                ''' % (host, port, host, port)))
            self.query('''SELECT max(sendmail(float1)) FROM test.enginetablebig1''') 

        data = mb.data
        self.assertGreater(len(data), 0)
        init = sorted([x.split(':')[1] for x in data if x.startswith('init')])
        cleanup = sorted([x.split(':')[1] for x in data if x.startswith('cleanup')])
        self.assertEquals(init, cleanup)
        # FIXME: math.random() is not thread-unique
        #self.assertEquals(sorted(set(init)), init)

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

