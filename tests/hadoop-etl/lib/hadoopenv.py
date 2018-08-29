#!/usr/bin/env python2.7
# encoding: utf8
import os

# Hadoop host info
host = 'hadoop01.omg.dev.exasol.com'
webhdfs_port = 50070
webhdfs_host = 'hadoop01.omg.dev.exasol.com:' + str(webhdfs_port)
webhcat_port = 50111
webhcat_host = 'hadoop01.omg.dev.exasol.com:' + str(webhcat_port)
user = 'hdfs'

# Hive info
hive_port = 10000
hive_schema = 'etl_udf'
hive_hdfs_path = '/tmp/import/'
hive_odbc_dir = '/usr/opt/cloudera/hiveodbc'
if not os.path.exists(hive_odbc_dir):
    hive_odbc_dir = '/opt/local/hiveodbc'

# Kerberos info
krb_dir = '/x/u/zg1089/hadoop-etl/hadoop_kerberos'
krb_conf = 'krb5.conf'
krb_keytab = 'hadoop.keytab'
krb_test_princ = 'hadooptester@OMG.DEV.EXASOL.COM'
krb_hdfs_princ = 'hdfs/_HOST@OMG.DEV.EXASOL.COM'
krb_http_princ = 'HTTP@hadoop01.omg.dev.exasol.com'
krb_realm = 'OMG.DEV.EXASOL.COM'
krb_hive_service_name = 'hive'
