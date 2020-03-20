import pathlib

from exaslct_src.lib.config.log_config import WriteLogFilesToConsole
from exaslct_src.lib.logging.abstract_log_handler import AbstractLogHandler


class CommandLogHandler(AbstractLogHandler):

    def __init__(self, log_file_path: pathlib.Path, logger, description: str):
        super().__init__(log_file_path, logger)
        self._description = description

    def handle_log_line(self, log_line, error: bool = False):
        log_line = log_line.decode("utf-8")
        self._log_file.write(log_line)
        self._complete_log.append(log_line)
        if error:
            self._error_message=log_line

    def finish(self):
        if self._log_config.write_log_files_to_console==WriteLogFilesToConsole.all:
            self._logger.info("Command log for %s \n%s",
                              self._description,
                              "".join(self._complete_log))
        if self._error_message is not None:
            if self._log_config.write_log_files_to_console == WriteLogFilesToConsole.only_error:
                self._logger.error("Command failed %s failed\nCommand Log:\n%s",
                                  self._description,
                                  "\n".join(self._complete_log))
            raise Exception(
                "Error occured during %s. Received error \"%s\" ."
                "The whole log can be found in %s"
                % (self._description,
                   self._error_message,
                   self._log_file_path.absolute()),
                self._log_file_path.absolute())
