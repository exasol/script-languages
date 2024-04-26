#!/usr/bin/env python3

import os
from pathlib import Path
import sys
import zipfile
import requests
from requests.auth import HTTPBasicAuth

from exasol_python_test_framework import udf
from exasol_python_test_framework import docker_db_environment
from exasol_python_test_framework.udf import useData, expectedFailure

script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
java_udf_dir = script_dir / "resources/java_udf"
java_udf_jar = java_udf_dir / "target/test-udf-java-17.jar"

@udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
class JavaModules(udf.TestCase):
    env = None
    
    def setUp(self):
        self.query('CREATE SCHEMA JAVA_MODULES', ignore_errors=True)
        self.query('OPEN SCHEMA JAVA_MODULES')
        self.env = docker_db_environment.DockerDBEnvironment("JAVA_MODULE_TEST")


    def test_udf_jar_is_module(self):
        with zipfile.ZipFile(java_udf_jar, 'r') as zip:
            assert "module-info.class" in zip.namelist()

    def upload_to_bucketfs(self, path: Path) -> str:
        docker_db_container = self.env.get_docker_db_container()
        docker_db_ip = self.env.get_ip_address_of_container(docker_db_container)
        upload_url = f"http://{docker_db_ip}:6583/myudfs/{path.name}"
        username = "w"
        password = "write"
        r_upload = requests.put(upload_url, data=path.read_bytes(), auth=HTTPBasicAuth(username, password))
        r_upload.raise_for_status()
        return f"/buckets/bfsdefault/myudfs/{path.name}"

    def test_module_jar_udf(self):
        java_udf_jar_bucketfs_path = self.upload_to_bucketfs(java_udf_jar)
        self.query(udf.fixindent(f'''
                CREATE JAVA SCALAR SCRIPT JAVA_MODULE_TEST() RETURNS INT AS
                %scriptclass com.exasol.slc.testudf.Main;
                %jar {java_udf_jar_bucketfs_path};
                '''))
        rows = self.query("SELECT JAVA_MODULE_TEST()")
        self.assertEqual(rows[0][0], 17)


if __name__ == '__main__':
    udf.main()
