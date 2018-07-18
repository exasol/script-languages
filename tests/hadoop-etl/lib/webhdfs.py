#!/usr/opt/bs-python-2.7/bin/python
# encoding: utf8

import sys
import os
import re
import httplib
import json
import hadoopenv
import gss

sys.path.append(os.path.realpath(__file__ + '/../../lib'))

def create_file(host, user, hdfs_path, file_path, overwrite = True, use_kerberos = False):
    conn = httplib.HTTPConnection(host)
    url = '/webhdfs/v1' + hdfs_path
    url += '?op=create'
    url += '&overwrite=' + ('true' if overwrite else 'false')
    if not use_kerberos:
        url += '&user.name=' + user
    auth_header = {}
    if use_kerberos:
        gss_client = gss.GssClient(hadoopenv.krb_http_princ, hadoopenv.krb_test_princ)
        auth_header = gss_client.get_auth_header()
    conn.request('PUT', url, headers = auth_header)
    resp = conn.getresponse()
    if resp.status != 307:
        raise RuntimeError('Expected 307: Temporary Redirect, got ' + str(resp.status) + ': ' + resp.reason)
    loc = resp.getheader('location')
    loc = re.sub('^http://', '', loc)
    host2 = loc[0 : loc.index('/')]
    url = loc[loc.index('/') :]
    conn2 = httplib.HTTPConnection(host2)
    with open(file_path, 'r') as file:
        conn2.request('PUT', url, file, headers = (auth_header if auth_header else None))
    resp = conn2.getresponse()
    if resp.status != 201:
        raise RuntimeError('Expected 201: Created, got ' + str(resp.status) + ': ' + resp.reason)



def append_file(host, user, hdfs_path, file_path):
    conn = httplib.HTTPConnection(host)
    url = '/webhdfs/v1' + hdfs_path
    url += '?op=append'
    url += '&user.name=' + user
    conn.request('POST', url)
    resp = conn.getresponse()
    if resp.status != 307:
        raise RuntimeError('Expected 307: Temporary Redirect, got ' + str(resp.status) + ': ' + resp.reason)
    loc = resp.getheader('location')
    loc = re.sub('^http://', '', loc)
    host2 = loc[0 : loc.index('/')]
    url = loc[loc.index('/') :]
    conn2 = httplib.HTTPConnection(host2)
    with open(file_path, 'r') as file:
        conn2.request('POST', url, file)
    resp = conn2.getresponse()
    if resp.status != 200:
        raise RuntimeError('Expected 200: OK, got ' + str(resp.status) + ': ' + resp.reason)



def get_file_http_response_object(host, user, hdfs_path):
    conn = httplib.HTTPConnection(host)
    url = '/webhdfs/v1' + hdfs_path
    url += '?op=open'
    url += '&user.name=' + user
    conn.request('GET', url)
    resp = conn.getresponse()
    if resp.status != 307:
        raise RuntimeError('Expected 307: Temporary Redirect, got ' + str(resp.status) + ': ' + resp.reason)
    loc = resp.getheader('location')
    loc = re.sub('^http://', '', loc)
    host2 = loc[0 : loc.index('/')]
    url = loc[loc.index('/') :]
    conn2 = httplib.HTTPConnection(host2)
    conn2.request('GET', url)
    resp = conn2.getresponse()
    if resp.status != 200:
        raise RuntimeError('Expected 200: OK, got ' + str(resp.status) + ': ' + resp.reason)
    return resp



def get_file_data(host, user, hdfs_path):
    resp = get_file_http_response_object(host, user, hdfs_path)
    return resp.read()



def list_dir(host, user, schema, table, use_kerberos = False):
    conn = httplib.HTTPConnection(host)
    url = '/webhdfs/v1'
    url += '/user/hive/warehouse/%s/%s/' % (schema + '.db', table)
    url += '?op=liststatus'
    if not use_kerberos:
        url += '&user.name=' + user
    auth_header = {}
    if use_kerberos:
        gss_client = gss.GssClient(hadoopenv.krb_http_princ, hadoopenv.krb_test_princ)
        auth_header = gss_client.get_auth_header()
    conn.request('GET', url, headers = auth_header)
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError('Expected 200: OK, got ' + str(resp.status) + ': ' + resp.reason)
    json_obj = json.loads(resp.read())
    files = []
    for file in json_obj['FileStatuses']['FileStatus']:
        files.append(file['pathSuffix'])
    return files
