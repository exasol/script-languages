#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData, expectedFailure

class JAVA_NATIVE(udf.TestCase):

    def setUp(self):
        self.query('CREATE SCHEMA JAVA_NATIVE', ignore_errors=True)
        self.query('OPEN SCHEMA JAVA_NATIVE')

    def test_java_native_management(self):
        self.query(udf.fixindent('''
                CREATE JAVA SET SCRIPT JAVA_NATIVE_MANAGEMENT (a varchar(100)) EMITS (b varchar(10000)) AS
                import java.lang.management.ManagementFactory;

                class JAVA_NATIVE_MANAGEMENT {
                  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    final String jvmName = ManagementFactory.getRuntimeMXBean().getName();
                    ctx.emit(jvmName);
                  }
                }
                '''))
        rows = self.query('''
            SELECT JAVA_NATIVE_MANAGEMENT('');
            ''')

    def test_java_native_net(self):
        self.query(udf.fixindent('''
                CREATE JAVA SET SCRIPT JAVA_NATIVE_NET (a varchar(100)) EMITS (b varchar(10000)) AS
                import java.net.NetworkInterface;

                class JAVA_NATIVE_NET {
                  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    final String jvmName = NetworkInterface.getNetworkInterfaces().nextElement().getName();
                    ctx.emit(jvmName);
                  }
                }
                '''))
        rows = self.query('''
            SELECT JAVA_NATIVE_NET('');
            ''')

if __name__ == '__main__':
    udf.main()

