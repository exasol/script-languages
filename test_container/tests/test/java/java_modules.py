#!/usr/bin/env python3

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile
import requests
from requests.auth import HTTPBasicAuth

from exasol_python_test_framework import udf
from exasol_python_test_framework import docker_db_environment
from exasol_python_test_framework.udf import useData, expectedFailure
from exasol_python_test_framework.udf.udf_debug import UdfDebugger
from exasol_python_test_framework.exatest.utils import obj_from_json_file


script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
java_udf_dir = script_dir / "resources/java_udf"

@udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
class JavaModules(udf.TestCase):
    env = None
    java_udf_jar_java11 = None
    java_udf_jar_java17 = None

    additional_env_declarations = [("",), ("%env SCRIPT_OPTIONS_PARSER_VERSION=2;",)]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.java_udf_jar_java11 = self.build_java_udf(11)
        self.java_udf_jar_java17 = self.build_java_udf(17)

    def setUp(self):
        self.query('CREATE SCHEMA JAVA_MODULES', ignore_errors=True)
        self.query('OPEN SCHEMA JAVA_MODULES')
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
        env_info = obj_from_json_file("/environment_info.json")
        docker_db_ip = env_info.database_info.container_info.ip_address
        bucketfs_port = env_info.database_info.ports.bucketfs
        upload_url = f"http://{docker_db_ip}:{bucketfs_port}/myudfs/{path.name}"
        username = "w"
        password = "write"
        print(f"Trying to upload to {upload_url}")
        r_upload = requests.put(upload_url, data=path.read_bytes(), auth=HTTPBasicAuth(username, password))
        r_upload.raise_for_status()
        return f"/buckets/bfsdefault/myudfs/{path.name}"

    @useData(additional_env_declarations)
    def test_java_11_udf(self, additional_env_declaration):
        """
        Verify that a module JAR built for Java 11 with a module-info.class can be used in a UDF.
        We don't know the JRE version installed in the SLC, so we accept both 11 and 17.
        """
        assert self.get_jre_version(additional_env_declaration) in ["11", "17"]

    @useData(additional_env_declarations)
    def test_java_17_udf(self, additional_env_declaration):
        """
        Verify that a module JAR built for Java 17 can be used in a UDF.
        We don't know the JRE version installed in the SLC, so first check that it is 11 or 17.
        Exepct an error if the JRE version is not 17.
        """
        bucketfs_path = self.upload_to_bucketfs(self.java_udf_jar_java17)
        self.query(udf.fixindent(f'''
            CREATE JAVA SCALAR SCRIPT JAVA_17_UDF() RETURNS INT AS
            {additional_env_declaration}
            %scriptclass com.exasol.slc.testudf.Main;
            %jar {bucketfs_path};
            '''))
        if self.get_jre_version(additional_env_declaration) == "17":
            rows = self.query("SELECT JAVA_17_UDF()")
            return str(rows[0][0])
        else:
            with self.assertRaisesRegex(Exception, "UnsupportedClassVersionError: com/exasol/slc/testudf/Main has been compiled by a more recent version of the Java Runtime"):
                self.query("SELECT JAVA_17_UDF()")

    def get_jre_version(self, additional_env_declaration) -> str:
        """
        Get the SLC's JRE version by executing a Java 11 UDF that returns the JRE major version ("11" or "17").
        """
        bucketfs_path = self.upload_to_bucketfs(self.java_udf_jar_java11)
        self.query(udf.fixindent(f'''
                CREATE JAVA SCALAR SCRIPT JRE_VERSION() RETURNS INT AS
                {additional_env_declaration}
                %scriptclass com.exasol.slc.testudf.Main;
                %jar {bucketfs_path};
                '''))
        rows = self.query("SELECT JRE_VERSION()")
        return str(rows[0][0])
 

if __name__ == '__main__':
    udf.main()
