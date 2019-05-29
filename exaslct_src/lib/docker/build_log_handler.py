import json

import docker

from exaslct_src.lib.abstract_log_handler import AbstractLogHandler
from exaslct_src.lib.data.image_info import ImageInfo
from exaslct_src.lib.log_config import WriteLogFilesToConsole


class BuildLogHandler(AbstractLogHandler):

    def __init__(self, log_file_path, logger, task_id, image_info: ImageInfo):
        super().__init__(log_file_path, logger, task_id)
        self._image_info = image_info

    def handle_log_line(self, log_line, error: bool = False):
        log_line = log_line.decode("utf-8")
        self._log_file.write(log_line)
        log_line = log_line.strip('\r\n')
        self._complete_log.append(log_line)
        json_output = json.loads(log_line)
        if 'errorDetail' in json_output:
            self._error_message = json_output["errorDetail"]["message"]

    def finish(self):
        self.write_log_to_console_if_requested()
        self.handle_error()

    def write_log_to_console_if_requested(self):
        if self._log_config.write_log_files_to_console == WriteLogFilesToConsole.all:
            self._logger.info("Task %s: Build Log of image %s\n%s",
                              self._task_id,
                              self._image_info.get_target_complete_name(),
                              "\n".join(self._complete_log))

    def handle_error(self):
        if self._error_message is not None:
            self.write_error_log_to_console_if_requested()
            raise docker.errors.BuildError(
                "Error occurred during the build of the image %s. Received error \"%s\" ."
                "The whole log can be found in %s"
                % (self._image_info.get_target_complete_name(),
                   self._error_message,
                   self._log_file_path.absolute()),
                self._log_file_path.absolute())

    def write_error_log_to_console_if_requested(self):
        if self._log_config.write_log_files_to_console == WriteLogFilesToConsole.only_error:
            self._logger.error("Task %s: Build of image %s failed\nBuild Log:\n%s",
                               self._task_id,
                               self._image_info.get_target_complete_name(),
                               "\n".join(self._complete_log))