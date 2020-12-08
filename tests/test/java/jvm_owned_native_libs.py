#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData, expectedFailure

class JVMOwnedNativeLibsTest(udf.TestCase):

    def setUp(self):
        self.query('CREATE SCHEMA JAVA_NATIVE', ignore_errors=True)
        self.query('OPEN SCHEMA JAVA_NATIVE')

    def test_libmanagement(self):
        self.query(udf.fixindent('''
                CREATE JAVA SET SCRIPT JAVA_LIBMANAGEMENT (a varchar(100)) EMITS (b varchar(10000)) AS
                import java.lang.management.ManagementFactory;

                class JAVA_LIBMANAGEMENT {
                  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    final String jvmName = ManagementFactory.getRuntimeMXBean().getName();
                    ctx.emit(jvmName);
                  }
                }
                '''))
        rows = self.query('''
            SELECT JAVA_LIBMANAGEMENT('');
            ''')

    def test_libnet(self):
        self.query(udf.fixindent('''
                CREATE JAVA SET SCRIPT JAVA_LIBNET (a varchar(100)) EMITS (b varchar(10000)) AS
                import java.net.NetworkInterface;

                class JAVA_LIBNET {
                  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    final String jvmName = NetworkInterface.getNetworkInterfaces().nextElement().getName();
                    ctx.emit(jvmName);
                  }
                }
                '''))
        rows = self.query('''
            SELECT JAVA_LIBNET('');
            ''')

    def test_libzip(self):
        self.query(udf.fixindent('''
                CREATE JAVA SET SCRIPT JAVA_LIBZIP (a varchar(100)) EMITS (b varchar(10000)) AS
                import java.io.File;
                import java.io.FileInputStream;
                import java.io.FileOutputStream;
                import java.io.IOException;
                import java.util.zip.ZipEntry;
                import java.util.zip.ZipOutputStream;

                class JAVA_LIBZIP {
                  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    String sourceFile = "/etc/hosts";
                    FileOutputStream fos = new FileOutputStream("/tmp/compressed.zip");
                    ZipOutputStream zipOut = new ZipOutputStream(fos);
                    File fileToZip = new File(sourceFile);
                    FileInputStream fis = new FileInputStream(fileToZip);
                    ZipEntry zipEntry = new ZipEntry(fileToZip.getName());
                    zipOut.putNextEntry(zipEntry);
                    byte[] bytes = new byte[1024];
                    int length;
                    while((length = fis.read(bytes)) >= 0) {
                        zipOut.write(bytes, 0, length);
                    }
                    zipOut.close();
                    fis.close();
                    fos.close();
                    ctx.emit("SUCCESS");
                  }
                }
                '''))
        rows = self.query('''
            SELECT JAVA_LIBZIP('');
            ''')

    def test_libnio(self):
        self.query(udf.fixindent('''
                CREATE JAVA SET SCRIPT JAVA_LIBNIO (a varchar(100)) EMITS (b varchar(10000)) AS
                import java.io.FileInputStream;
                import java.io.FileOutputStream;
                import java.nio.ByteBuffer;
                import java.nio.channels.FileChannel;
                import java.nio.charset.StandardCharsets;


                class JAVA_LIBNIO {
                  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    FileInputStream fin= new FileInputStream("/etc/hosts");
                    FileChannel channel = fin.getChannel();
                    ByteBuffer buff = ByteBuffer.allocate(1024);
                    int noOfBytesRead = channel.read(buff);
                    String fileContent = new String(buff.array(), StandardCharsets.UTF_8);
                    channel.close();
                    fin.close();
                    ctx.emit("SUCCESS");
                  }
                }
                '''))
        rows = self.query('''
            SELECT JAVA_LIBNIO('');
            ''')

if __name__ == '__main__':
    udf.main()

