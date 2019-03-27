import time

import docker
import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.data.container_info import ContainerInfo
from build_utils.lib.data.database_info import DatabaseInfo
from build_utils.lib.docker_config import docker_config


class UploadFileToDB(luigi.Task):
    database_info_dict = luigi.DictParameter()
    test_container_info_dict = luigi.DictParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._build_config = build_config()
        self.test_container_info = ContainerInfo.from_dict(self.test_container_info_dict)
        self.database_info = DatabaseInfo.from_dict(self.database_info_dict)
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._log_target = luigi.LocalTarget(
            "%s/logs/test-runner/db-test/upload/%s/%s"
            % (self._build_config.ouput_directory,
               self.test_container_info.container_name,
               self.task_id))
        if self._log_target.exists():
            self._log_target.remove()

    def output(self):
        return self._log_target

    def run(self):
        file_to_upload = self.get_file_to_upload()
        upload_target = self.get_upload_target()
        pattern_to_wait_for = self.get_pattern_to_wait_for()
        log_file = self.get_log_file()

        self.upload_file(file_to_upload=file_to_upload, upload_target=upload_target)
        self.wait_for_upload(pattern_to_wait_for=pattern_to_wait_for, log_file=log_file)
        self.write_logs("Done".encode("utf-8"))

    def get_log_file(self) -> str:
        pass

    def get_pattern_to_wait_for(self) -> str:
        pass

    def get_file_to_upload(self) -> str:
        pass

    def get_upload_target(self) -> str:
        pass

    def upload_file(self, file_to_upload: str, upload_target: str):
        test_container = self._client.containers.get(self.test_container_info.container_name)
        url = "http://w:write@{host}:{port}/{target}".format(
            host=self.database_info.host, port=self.database_info.bucketfs_port,
            target=upload_target)
        cmd = "curl -v -X PUT -T {jar} {url}".format(jar=file_to_upload, url=url)
        exit_code, output = test_container.exec_run(cmd=cmd)
        if exit_code != 0:
            self.write_logs(output)
            raise Exception("Upload of %s failed" % file_to_upload)

    def wait_for_upload(self, pattern_to_wait_for: str, log_file: str):
        database_container = self._client.containers.get(self.database_info.container_info.container_name)
        exit_code = 1
        i = 0
        while exit_code != 0:
            cmd = f"""grep '{pattern_to_wait_for}' {log_file} """
            bash_cmd = f"""bash -c "{cmd}" """
            exit_code, output = \
                database_container.exec_run(cmd=bash_cmd)
            i += 1
            time.sleep(1)

    def write_logs(self, output):
        with self._log_target.open("w") as file:
            file.write(output.decode("utf-8"))
