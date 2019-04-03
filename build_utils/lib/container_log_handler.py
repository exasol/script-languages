from build_utils.lib.abstract_log_handler import AbstractLogHandler


class ContainerLogHandler(AbstractLogHandler):

    def __init__(self, log_file_path, logger, task_id, description:str):
        super().__init__(log_file_path, logger, task_id)
        self.db_container_name = description

    def handle_log_line(self, log_line, error:bool=False):
        log_line = log_line.decode("utf-8")
        self._log_file.write(log_line)
        self._complete_log.append(log_line)

    def finish(self):
        if self._log_config.write_log_files_to_console:
            self._logger.info("Task %s: Log %s\n%s",
                              self._task_id,
                              self.db_container_name,
                              "\n".join(self._complete_log))