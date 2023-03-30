#!/usr/bin/env python2.7
# encoding: utf8

import sys
import os
import hadoopenv

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))
from exatest.clients.odbc import ODBCClient

class HiveOdbc(ODBCClient):
    def __init__(self, use_kerberos):
        dsn = 'Cloudera Hive'
        super(HiveOdbc, self).__init__(dsn)
        self.params = {}
        self.params['dsn'] = dsn
        self.params['autocommit'] = True
        self.use_krb = use_kerberos
        if os.path.isfile(os.environ['ODBCINI']):
            with open(os.environ['ODBCINI'], 'r') as f:
                if dsn not in f.read():
                    # Append Hive DSN if not present
                    self._append_odbc_ini(dsn)
        else:
            raise RuntimeError('odbc.ini does not exist')
        self._write_cloudera_init()

    def _append_odbc_ini(self, dsn):
        with open(os.environ['ODBCINI'], 'r') as f:
            odbcini = f.read()
        lines = odbcini.split('\n')
        for idx, line in enumerate(lines):
            if line == '[ODBC Data Sources]':
                lines.insert(idx + 1, '%s=%s' % (dsn, dsn))
                break
        lines += self._build_hive_odbc_ini()
        with open(os.environ['ODBCINI'], 'w') as f:
            for line in lines:
                f.write(line + '\n')

    def _build_hive_odbc_ini(self):
        file = 'odbc.ini'
        cloudera_path = hadoopenv.hive_odbc_dir + '/Setup/' + file
        with open(cloudera_path, 'r') as f:
            cloudera_ini = f.read()
        keep_reached = False
        lines = cloudera_ini.split('\n')
        data_source_keep = '[Sample Cloudera Hive DSN 64]'
        out_lines = []
        for line in lines:
            if data_source_keep in line:
                keep_reached = True
            # Rename DSN
            if data_source_keep.strip('[]') in line:
                line = line.replace(data_source_keep.strip('[]'), self.params['dsn'])
            if keep_reached:
                if 'HOST=' in line:
                    line = 'HOST=' + hadoopenv.host
                elif 'PORT=' in line:
                    line = 'PORT=' + str(hadoopenv.hive_port)
                elif 'Schema=' in line:
                    line = 'Schema=' + hadoopenv.hive_schema
                elif 'UID=' in line:
                    line = 'UID=' + hadoopenv.user
                elif 'Driver=' in line:
                    driver = line[line.rfind('/') + 1 :]
                    line = 'Driver=' + hadoopenv.hive_odbc_dir + '/lib/64/' + driver
                elif self.use_krb and 'AuthMech=' in line:
                    line = 'AuthMech=1'
                elif self.use_krb and 'KrbHostFQDN=' in line:
                    line = 'KrbHostFQDN=' + hadoopenv.host
                elif self.use_krb and 'KrbServiceName=' in line:
                    line = 'KrbServiceName=' + hadoopenv.krb_hive_service_name
                elif self.use_krb and 'KrbRealm=' in line:
                    line = 'KrbRealm=' + hadoopenv.krb_realm
                out_lines.append(line)
        return out_lines

    def _write_cloudera_init(self):
        file = 'cloudera.hiveodbc.ini'
        cloudera_path = hadoopenv.hive_odbc_dir + '/lib/64/' + file
        test_path = os.path.realpath(os.path.join('.', file))
        with open(cloudera_path, 'r') as f:
            cloudera_ini = f.read()
        lines = cloudera_ini.split('\n')
        unixodbc = 'SimbaDM / unixODBC'
        found_unixodbc = False
        with open(test_path, 'w') as f:
            for line in lines:
                # Change ODBCInstLib to unixODBC
                if unixodbc in line:
                    found_unixodbc = True
                elif found_unixodbc:
                    eq_idx = line.find('=')
                    line = line[1 : eq_idx + 1] + line[eq_idx + 1 :]
                    found_unixodbc = False
                elif line.startswith('ODBCInstLib='):
                    line = '#' + line
                # Change encoding
                elif 'DriverManagerEncoding=' in line:
                    line = 'DriverManagerEncoding=UTF-16'
                # Change error messages dir
                elif 'ErrorMessagesPath=' in line:
                    line = 'ErrorMessagesPath=' + hadoopenv.hive_odbc_dir + '/ErrorMessages/'
                f.write(line + '\n')
        os.environ['CLOUDERAHIVEINI'] = test_path

    def commit(self):
        raise RuntimeError('Hive does not support transactions')

    def rollback(self):
        raise RuntimeError('Hive does not support transactions')
