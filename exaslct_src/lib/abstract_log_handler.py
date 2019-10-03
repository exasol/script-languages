from exaslct_src.lib.log_config import log_config


class AbstractLogHandler:

    def __init__(self, log_file_path, logger):
        self._log_file_path = log_file_path
        self._logger = logger
        self._complete_log = []
        self._error_message = None
        self._log_config = log_config()

    def __enter__(self):
        self._log_file = self._log_file_path.open("w")
        return self

    def __exit__(self, type, value, traceback):
        self._log_file.close()
        self.finish()

    def handle_log_line(self, log_line, error:bool=False):
        pass

    def finish(self):
        pass