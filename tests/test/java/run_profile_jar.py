#!/usr/bin/env python3
import datetime

from exasol_python_test_framework import udf
from exasol_python_test_framework import docker_db_environment
from exasol_python_test_framework.udf.udf_debug import UdfDebugger
from requests.auth import HTTPBasicAuth


class JarJavaTest(udf.TestCase):
    def setUp(self):
        self.query('CREATE SCHEMA FN2', ignore_errors=True)
        self.query('OPEN SCHEMA FN2')

    def upload_profiling_jar_to_bucket_fs(self, env):
        docker_db_container = env.get_docker_db_container()
        docker_db_ip = env.get_ip_address_of_container(docker_db_container)
        upload_url = "http://{docker_db_ip}:6583/myudfs/ProfilingUdf-1.0-SNAPSHOT.jar".format(
            docker_db_ip=docker_db_ip)
        username = "w"
        password = "write"  # TOOD hardcoded
        import requests
        with open("/ProfilingUdf-1.0-SNAPSHOT.jar", 'rb') as profiling_jar_file:
            r_upload = requests.put(upload_url, data=profiling_jar_file.read(),
                                auth=HTTPBasicAuth(username, password))
            r_upload.raise_for_status()

    def create_profiling_udfs(self):
        self.query(udf.fixindent("""
                CREATE OR REPLACE java SCALAR SCRIPT
                run_it()
                RETURNS int AS
                %scriptclass com.exasol.udf_profiling.UdfProfiler;
                %jar /buckets/bfsdefault/myudfs/ProfilingUdf-1.0-SNAPSHOT.jar;
                /
                """))

    def test_simple(self):
        env = docker_db_environment.DockerDBEnvironment("JAVA_PROFILE")
        self.upload_profiling_jar_to_bucket_fs(env)
        self.create_profiling_udfs()
        with UdfDebugger(test_case=self):
            ct = datetime.datetime.now()
            print(f"PROFILING[BEGIN SQL QUERY] {ct.hour}:{ct.minute}:{ct.second}.{round(ct.microsecond/1000)}")
            row = self.query('SELECT run_it() FROM DUAL')[0]
            print(f"PROFILING[END SQL QUERY] {ct.hour}:{ct.minute}:{ct.second}.{round(ct.microsecond / 1000)}")


if __name__ == '__main__':
    udf.main()

