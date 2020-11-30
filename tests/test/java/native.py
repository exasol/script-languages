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

    def test_java_loadlibrary_net(self):
        self.query(udf.fixindent('''
                CREATE JAVA SET SCRIPT JAVA_LOADLIBRARY_NET (a varchar(100)) EMITS (b varchar(10000)) AS
import java.security.AccessController;
import java.io.IOException;
import java.security.Principal;
import javax.security.auth.Subject;
import javax.security.auth.login.LoginContext;
import java.security.PrivilegedAction;
class JAVA_LOADLIBRARY_NET {


    static class User implements Principal {
      private final String fullName;

      public User(String name) {
        fullName = name;
      }

      /**
       * Get the full name of the user.
       */
      @Override
      public String getName() {
        return fullName;
      }
      
      
      @Override
      public boolean equals(Object o) {
        if (this == o) {
          return true;
        } else if (o == null || getClass() != o.getClass()) {
          return false;
        } else {
          return ((fullName.equals(((User) o).fullName)));
        }
      }
      
      @Override
      public int hashCode() {
        return fullName.hashCode();
      }
      
      @Override
      public String toString() {
        return fullName;
      }

      
    }



    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        ctx.emit(System.mapLibraryName("net"));
/*
        Subject subject = new Subject();
        User userEntry = new User("hive");
        subject.getPrincipals().add(userEntry);
       PrivilegedAction<Void> load_library = 
	                    new PrivilegedAction<Void>() {
	                        public Void run() {
	                            System.loadLibrary("net");
	                            return null;
	                        }
	                    };
        PrivilegedAction<Void> do_privileged = 
            new PrivilegedAction<Void>() {
	            public Void run() {
	                AccessController.doPrivileged(load_library);
	                return null;
	            }
            };
        Subject.doAs(subject, do_privileged);
*/
    }
}
/
                '''))
        rows = self.query('''
            SELECT JAVA_LOADLIBRARY_NET('');
            ''')
        print(rows)
if __name__ == '__main__':
    udf.main()

