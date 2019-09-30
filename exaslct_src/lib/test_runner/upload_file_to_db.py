import logging

import luigi
from docker.models.containers import Container

from exaslct_src.AbstractMethodException import AbstractMethodException
from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.environment_info import EnvironmentInfo
from exaslct_src.lib.docker_config import docker_client_config
from exaslct_src.lib.still_running_logger import StillRunningLoggerThread, StillRunningLogger
from exaslct_src.lib.stoppable_task import StoppableTask
from exaslct_src.lib.test_runner.docker_db_log_based_bucket_sync_checker import DockerDBLogBasedBucketFSSyncChecker
from exaslct_src.lib.test_runner.time_based_bucketfs_sync_waiter import TimeBasedBucketFSSyncWaiter


# TODO add timeout, because sometimes the upload stucks
class UploadFileToBucketFS(StoppableTask):
    logger = logging.getLogger('luigi-interface')

    environment_name = luigi.Parameter()
    test_environment_info_dict = luigi.DictParameter(significant=False)
    reuse_uploaded = luigi.BoolParameter(False, significant=False)
    bucketfs_write_password = luigi.Parameter(significant=False, visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._client = docker_client_config().get_client()

        self._test_environment_info = test_environment_info = EnvironmentInfo.from_dict(self.test_environment_info_dict)
        self._test_container_info = test_environment_info.test_container_info
        self._database_info = test_environment_info.database_info
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._log_target = luigi.LocalTarget(
            "%s/logs/environment/%s/bucketfs-upload/%s"
            % (build_config().output_directory,
               self._test_environment_info.name,
               self.task_id))
        if self._log_target.exists():
            self._log_target.remove()

    def output(self):
        return self._log_target

    def run_task(self):
        file_to_upload = self.get_file_to_upload()
        upload_target = self.get_upload_target()
        pattern_to_wait_for = self.get_pattern_to_wait_for()
        log_file = self.get_log_file()
        sync_time_estimation = self.get_sync_time_estimation()

        if self._database_info.container_info is not None:
            database_container = self._client.containers.get(
                self._database_info.container_info.container_name)
        else:
            database_container = None
        if not self.should_be_reused(upload_target):
            self.upload_and_wait(database_container,
                                 file_to_upload,
                                 upload_target,
                                 log_file,
                                 pattern_to_wait_for,
                                 sync_time_estimation)
        else:
            self.logger.warning("Task %s: Reusing uploaded target %s instead of file %s",
                                self.__repr__(), upload_target, file_to_upload)
            self.write_logs("Reusing")

    def upload_and_wait(self, database_container,
                        file_to_upload:str, upload_target:str,
                        log_file:str, pattern_to_wait_for:str,
                        sync_time_estimation:int):
        still_running_logger = StillRunningLogger(self.logger, self.__repr__(),
                                                  "file upload of %s to %s"
                                                  % (file_to_upload, upload_target))
        thread = StillRunningLoggerThread(still_running_logger)
        thread.start()
        sync_checker = self.get_sync_checker(database_container, sync_time_estimation,
                                             log_file, pattern_to_wait_for)
        sync_checker.prepare_upload()
        output = self.upload_file(file_to_upload=file_to_upload, upload_target=upload_target)
        sync_checker.wait_for_bucketfs_sync()
        thread.stop()
        thread.join()
        self.write_logs(output)

    def get_sync_checker(self, database_container:Container,
                         sync_time_estimation: int,
                         log_file:str,
                         pattern_to_wait_for:str):
        if database_container is not None:
            return DockerDBLogBasedBucketFSSyncChecker(
                database_container=database_container,
                log_file_to_check=log_file,
                pattern_to_wait_for=pattern_to_wait_for,
                task_id=self.__repr__(),
                bucketfs_write_password=self.bucketfs_write_password
            )
        else:
            return TimeBasedBucketFSSyncWaiter(sync_time_estimation)


    def should_be_reused(self, upload_target: str):
        return self.reuse_uploaded and self.exist_file_in_bucketfs(upload_target)

    def exist_file_in_bucketfs(self, upload_target: str) -> bool:
        self.logger.info("Task %s: Check if file %s exist in bucketfs", self.__repr__(),
                         upload_target)
        command = self.generate_list_command(upload_target)
        exit_code, log_output = self.run_command("list", command)

        if exit_code != 0:
            self.write_logs(log_output)
            raise Exception("Task %s: List files in bucketfs failed, got following output %s"
                            % (self.task_id, log_output))
        upload_target_in_bucket="/".join(upload_target.split("/")[1:])
        if upload_target_in_bucket in log_output.splitlines():
            return True
        else:
            return False

    def generate_list_command(self, upload_target: str):
        bucket = upload_target.split("/")[0]
        url = "http://w:{password}@{host}:{port}/{bucket}".format(
            host=self._database_info.host, port=self._database_info.bucketfs_port,
            bucket=bucket, password=self.bucketfs_write_password)
        cmd = f"curl --fail '{url}'"
        return cmd

    def upload_file(self, file_to_upload: str, upload_target: str):
        self.logger.info("Task %s: upload file %s to %s", self.__repr__(),
                         file_to_upload, upload_target)
        command = self.generate_upload_command(file_to_upload, upload_target)
        exit_code, log_output = self.run_command("upload", command)
        if exit_code != 0:
            self.write_logs(log_output)
            raise Exception("Task %s: Upload of %s failed, got following output %s"
                            % (self.task_id, file_to_upload, log_output))
        return log_output

    def generate_upload_command(self, file_to_upload, upload_target):
        url = "http://w:{password}@{host}:{port}/{target}".format(
            host=self._database_info.host, port=self._database_info.bucketfs_port,
            target=upload_target, password=self.bucketfs_write_password)
        cmd = f"curl --fail -X PUT -T '{file_to_upload}' '{url}'"
        return cmd

    def run_command(self, command_type, cmd):
        test_container = self._client.containers.get(self._test_container_info.container_name)
        self.logger.info("Task %s: start %s command %s", self.__repr__(), command_type, cmd)
        exit_code, output = test_container.exec_run(cmd=cmd)
        self.logger.info("Task %s: finish %s command %s", self.__repr__(), command_type, cmd)
        log_output = cmd + "\n\n" + output.decode("utf-8")
        return exit_code, log_output

    def write_logs(self, output):
        with self._log_target.open("w") as file:
            file.write(output)

    def get_log_file(self) -> str:
        raise AbstractMethodException()

    def get_pattern_to_wait_for(self) -> str:
        raise AbstractMethodException()

    def get_file_to_upload(self) -> str:
        raise AbstractMethodException()

    def get_upload_target(self) -> str:
        raise AbstractMethodException()

    def get_sync_time_estimation(self) -> int:
        """Estimated time in seconds which the bucketfs needs to extract and sync a uploaded file"""
        raise AbstractMethodException()
