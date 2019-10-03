from typing import List

from exaslct_src.lib.abstract_log_handler import AbstractLogHandler
from exaslct_src.lib.log_config import WriteLogFilesToConsole


class ContainerLogHandler(AbstractLogHandler):

    def __init__(self, log_file_path, logger, description: str):
        super().__init__(log_file_path, logger)
        self.db_container_name = description

    def handle_log_line(self, log_line, error: bool = False):
        log_line = log_line.decode("utf-8")
        self._log_file.write(log_line)
        self._complete_log.append(log_line)

    def get_complete_log(self) -> List[str]:
        return self._complete_log

    def finish(self):
        if self._log_config.write_log_files_to_console == WriteLogFilesToConsole.all:
            self._logger.info("Log %s\n%s", self.db_container_name,
                              "\n".join(self._complete_log))
