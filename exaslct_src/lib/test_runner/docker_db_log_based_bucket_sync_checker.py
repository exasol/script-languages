import logging
import time
from datetime import datetime

from docker.models.containers import Container

from exaslct_src.lib.test_runner.bucketfs_sync_checker import BucketFSSyncChecker


class DockerDBLogBasedBucketFSSyncChecker(BucketFSSyncChecker):

    def __init__(self, logger,
                 database_container: Container,
                 pattern_to_wait_for: str,
                 log_file_to_check: str,
                 bucketfs_write_password: str):
        self.logger = logger
        self.pattern_to_wait_for = pattern_to_wait_for
        self.log_file_to_check = log_file_to_check
        self.database_container = database_container
        self.bucketfs_write_password = bucketfs_write_password

    def prepare_upload(self):
        self.start_exit_code, self.start_output = self.find_pattern_in_logfile()

    def wait_for_bucketfs_sync(self):
        self.logger.info("wait for upload of file")

        ready = False
        while not ready:
            exit_code, output = self.find_pattern_in_logfile()
            if self.exit_code_changed(exit_code, self.start_exit_code) or \
                    self.found_new_log_line(exit_code, self.start_exit_code,
                                            self.start_output, output):
                ready = True
            time.sleep(1)

    def exit_code_changed(self, exit_code, start_exit_code):
        return exit_code == 0 and start_exit_code != 0

    def found_new_log_line(self, exit_code, start_exit_code,
                           start_output, output):
        return exit_code == 0 and start_exit_code == 0 and len(start_output) < len(output)

    def find_pattern_in_logfile(self):
        cmd = f"""grep '{self.pattern_to_wait_for}' {self.log_file_to_check}"""
        bash_cmd = f"""bash -c "{cmd}" """
        exit_code, output = \
            self.database_container.exec_run(bash_cmd)
        return exit_code, output

    def output_happened_after_start_time(self, output, start_time):
        time_str_from_output = " ".join(output.decode("utf-8").split(" ")[1:3])
        time_from_output = datetime.strptime(time_str_from_output, "%y%m%d %H:%M:%S")
        happened_after = time_from_output > start_time
        return happened_after