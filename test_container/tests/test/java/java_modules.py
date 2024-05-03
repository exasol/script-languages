#!/usr/bin/env python3

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
import requests
from requests.auth import HTTPBasicAuth

from exasol_python_test_framework import udf
from exasol_python_test_framework import docker_db_environment
from exasol_python_test_framework.udf import useData, expectedFailure

script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
java_udf_dir = script_dir / "resources/java_udf"

@udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
class JavaModules(udf.TestCase):
    env = None
    java_udf_jar_java11 = None
    java_udf_jar_java17 = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.java_udf_jar_java11 = self.build_java_udf(11)
        self.java_udf_jar_java17 = self.build_java_udf(17)

    def setUp(self):
        self.query('CREATE SCHEMA JAVA_MODULES', ignore_errors=True)
        self.query('OPEN SCHEMA JAVA_MODULES')
        self.query("ALTER SESSION SET SCRIPT_OUTPUT_ADDRESS='192.168.179.38:3000'")
        self.env = docker_db_environment.DockerDBEnvironment("JAVA_MODULE_TEST")

    def build_java_udf(self, javaVersion: int) -> Path:
        result = subprocess.run(["mvn", "--batch-mode", "--file", java_udf_dir/"pom.xml", "clean", "package", f"-Djava.version={javaVersion}"], stdout=subprocess.PIPE)
        result.check_returncode()
        jar = java_udf_dir / f"target/test-udf-java-{javaVersion}.jar"
        assert jar.exists()
        target = Path(tempfile.gettempdir()) / f"test-udf-java-{javaVersion}.jar"
        shutil.move(jar, target)
        return target

    def test_udf_jar_is_module(self):
        with zipfile.ZipFile(self.java_udf_jar_java11, 'r') as zip:
            assert "module-info.class" in zip.namelist()
    
    def test_udf_jar_is_module_java17(self):
        with zipfile.ZipFile(self.java_udf_jar_java17, 'r') as zip:
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

    def test_module_jar_udf_classpath(self):
        java_udf_jar_bucketfs_path = self.upload_to_bucketfs(self.java_udf_jar_java17)
        self.query(udf.fixindent(f'''
                CREATE JAVA SCALAR SCRIPT JAVA_MODULE_TEST() RETURNS INT AS
                %scriptclass com.exasol.slc.testudf.Main;
                %jar {java_udf_jar_bucketfs_path};
                '''))
        rows = self.query("SELECT JAVA_MODULE_TEST()")
        self.assertEqual(rows[0][0], 17)
        
    def test_module_jar_udf_modulepath_fails(self):
        java_udf_jar_bucketfs_path = self.upload_to_bucketfs(self.java_udf_jar_java17)
        self.query(udf.fixindent(f'''
                CREATE JAVA SCALAR SCRIPT JAVA_MODULE_TEST() RETURNS INT AS
                %scriptclass com.exasol.slc.testudf.Main;
                %jvmoption --module-path {java_udf_jar_bucketfs_path};
                '''))
        with self.assertRaisesRegex(Exception, "VM error: F-UDF-CL-LIB-1125: F-UDF-CL-SL-JAVA-1000: F-UDF-CL-SL-JAVA-1028: Cannot start the JVM: unknown error \\(-1\\)"):
            self.query("SELECT JAVA_MODULE_TEST()")

if __name__ == '__main__':
    udf.main()
