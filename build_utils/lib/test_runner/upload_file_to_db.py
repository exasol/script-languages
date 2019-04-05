import logging
import time
from datetime import datetime

import docker
import luigi
from docker.models.containers import Container

from build_utils.lib.build_config import build_config
from build_utils.lib.data.environment_info import EnvironmentInfo
from build_utils.lib.docker_config import docker_config
from build_utils.lib.still_running_logger import StillRunningLoggerThread, StillRunningLogger


class UploadFileToBucketFS(luigi.Task):
    logger = logging.getLogger('luigi-interface')

    environment_name = luigi.Parameter()
    test_environment_info_dict = luigi.DictParameter(significant=False)
    reuse_uploaded = luigi.BoolParameter(False, significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._build_config = build_config()
        test_environment_info = EnvironmentInfo.from_dict(self.test_environment_info_dict)
        self._test_container_info = test_environment_info.test_container_info
        self._database_info = test_environment_info.database_info
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._log_target = luigi.LocalTarget(
            "%s/logs/test-runner/db-test/bucketfs-upload/%s/%s"
            % (self._build_config.ouput_directory,
               self._test_container_info.container_name,
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
        database_container = self._client.containers.get(
            self._database_info.container_info.container_name)
        if not self.should_be_reused(database_container, log_file, pattern_to_wait_for):
            output = self.upload_and_wait(database_container, file_to_upload, log_file,
                                          pattern_to_wait_for, upload_target)
            self.write_logs(output)
        else:
            self.logger.warning("Task %s: Reusing uploaded target %s instead of file %s",
                                self.task_id, upload_target, file_to_upload)
            self.write_logs("Reusing")

    def should_be_reused(self, database_container: Container, log_file: str,
                         pattern_to_wait_for: str):
        if self.reuse_uploaded:
            exit_code, output = self.find_pattern_in_logfile(
                database_container=database_container,
                log_file=log_file,
                pattern_to_wait_for=pattern_to_wait_for)
            return exit_code == 0
        else:
            return False

    def upload_and_wait(self, database_container: Container, file_to_upload: str,
                        log_file: str, pattern_to_wait_for: str, upload_target: str):
        still_running_logger = StillRunningLogger(self.logger, self.task_id,
                                                  "file upload of %s to %s"
                                                  % (file_to_upload, upload_target))
        thread = StillRunningLoggerThread(still_running_logger)
        thread.start()
        utc_now = datetime.utcnow()
        output = self.upload_file(file_to_upload=file_to_upload, upload_target=upload_target)
        self.wait_for_upload(
            database_container=database_container,
            pattern_to_wait_for=pattern_to_wait_for,
            log_file=log_file, start_time=utc_now)
        thread.stop()
        thread.join()
        return output

    def get_log_file(self) -> str:
        pass

    def get_pattern_to_wait_for(self) -> str:
        pass

    def get_file_to_upload(self) -> str:
        pass

    def get_upload_target(self) -> str:
        pass

    def upload_file(self, file_to_upload: str, upload_target: str):
        self.logger.info("Task %s: upload file %s to %s", self.task_id,
                         file_to_upload, upload_target)
        test_container = self._client.containers.get(self._test_container_info.container_name)
        url = "http://w:write@{host}:{port}/{target}".format(
            host=self._database_info.host, port=self._database_info.bucketfs_port,
            target=upload_target)
        cmd = "curl -v -X PUT -T {jar} {url}".format(jar=file_to_upload, url=url)
        exit_code, output = test_container.exec_run(cmd=cmd)
        log_output=cmd+"\n\n"+output.decode("utf-8")
        if exit_code != 0:
            self.write_logs(log_output)
            raise Exception("Upload of %s failed, got following output %s"
                            % file_to_upload, output.decode("utf-8"))
        return log_output

    def wait_for_upload(self,
                        database_container: Container,
                        pattern_to_wait_for: str,
                        log_file: str,
                        start_time: datetime):
        self.logger.info("Task %s: wait for upload of file", self.task_id)
        ready = False
        i = 0
        while not ready:
            exit_code, output = self.find_pattern_in_logfile(
                database_container, log_file, pattern_to_wait_for)
            if exit_code == 0 and output != b'':
                ready = self.output_happend_after_start_time(output, start_time)
            i += 1
            time.sleep(1)

    def find_pattern_in_logfile(self, database_container: Container,
                                log_file: str, pattern_to_wait_for: str):
        cmd = f"""grep -B 0 -A 0 '{pattern_to_wait_for}' {log_file} | tail -1 """
        bash_cmd = f"""bash -c "{cmd}" """
        exit_code, output = \
            database_container.exec_run(cmd=bash_cmd)
        return exit_code, output

    def output_happend_after_start_time(self, output, start_time):
        time_str_from_output = " ".join(output.decode("utf-8").split(" ")[1:3])
        time_from_output = datetime.strptime(time_str_from_output, "%y%m%d %H:%M:%S")
        happend_after = time_from_output > start_time
        return happend_after

    def write_logs(self, output):
        with self._log_target.open("w") as file:
            file.write(output)
