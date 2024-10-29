import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import tarfile

import requests
from exasol_python_test_framework import udf
from exasol_python_test_framework import docker_db_environment
from exasol_python_test_framework.udf import useData, expectedFailure
from exasol_python_test_framework.udf.udf_debug import UdfDebugger
from exasol_python_test_framework.exatest.utils import obj_from_json_file
from requests.auth import HTTPBasicAuth

script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
java_udf_dir = script_dir / "resources/java_udf"

@udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
class JavaExtJarFileReferences(udf.TestCase):
    """
    Test for the Script Options Parser v2 which validates that correct processing of the (%jar) option.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.java_udf_jar_java = self.build_java_udf()

    def setUp(self):
        self.query('CREATE SCHEMA JAVA_MODULES', ignore_errors=True)
        self.query('OPEN SCHEMA JAVA_MODULES')
        self.env = docker_db_environment.DockerDBEnvironment("JAVA_MODULE_TEST")
        self.jar_target = self.build_java_udf()

    def build_java_udf(self) -> Path:
        result = subprocess.run(["mvn", "--batch-mode", "--file", java_udf_dir / "pom.xml", "clean", "package",
                                 f"-Djava.version=11"], stdout=subprocess.PIPE)
        result.check_returncode()
        jar = java_udf_dir / f"target/test-udf-java-11.jar"
        assert jar.exists()
        target = Path(tempfile.gettempdir()) / f"test-udf-java-11.jar"
        shutil.move(jar, target)
        return target

    def upload_to_bucketfs(self, src_path: Path, expected_jar_in_script_option: str, target_filename: str) -> str:
        env_info = obj_from_json_file("/environment_info.json")
        docker_db_ip = env_info.database_info.container_info.ip_address
        bucketfs_port = env_info.database_info.ports.bucketfs
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_target_file_path = Path(tmp_dir) / target_filename
            shutil.copy(src_path, local_target_file_path)
            target_targz_filename = Path(tmp_dir) / "java_jar_ref_test.tar.gz"
            with tarfile.open(target_targz_filename, "w:gz") as tar:
                tar.add(local_target_file_path, arcname=target_filename)
            upload_url = f"http://{docker_db_ip}:{bucketfs_port}/myudfs/jar_references_test/java_jar_ref_test.tar.gz"
            username = "w"
            password = "write"
            print(f"Trying to upload to {upload_url}")
            r_upload = requests.put(upload_url, data=target_targz_filename.read_bytes(), auth=HTTPBasicAuth(username, password))
            r_upload.raise_for_status()
        return f"/buckets/bfsdefault/myudfs/jar_references_test/java_jar_ref_test/{expected_jar_in_script_option}"

    @useData([("java_jar.jar", "java_jar.jar",1), ("java_jar.jar\ ", "java_jar.jar ", 2),
              (r"java_jar.jar\  ", r"java_jar.jar ", 3), ("java_jar.jar\\t", "java_jar.jar\t", 4),
              ("java_jar.jar ", "java_jar.jar", 5)])
    def test_jar_references(self, expected_jar_in_script_option, uploaded_jar, idx):
        """
        Install the jar file with the specific file name in BucketFS, and then create the UDF
        which uses this JAR file name. The JAR filename in the UDF (%jar) needs to be properly encoded.
        """
        bucketfs_path = self.upload_to_bucketfs(self.jar_target, expected_jar_in_script_option, uploaded_jar)
        self.query(udf.fixindent(f'''
            CREATE JAVA SCALAR SCRIPT JAVA_TEST_JAR_UDF_{idx}() RETURNS INT AS
            %scriptclass com.exasol.slc.testudf.Main;
            %env SCRIPT_OPTIONS_PARSER_VERSION=2;
            %jar {bucketfs_path};
            '''))
        rows = self.query(f"SELECT JAVA_TEST_JAR_UDF_{idx}()")
        self.assertGreaterEqual(rows[0][0], 11)


if __name__ == '__main__':
    udf.main()
