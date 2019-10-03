import math
import time
from pathlib import Path
from threading import Thread

from docker.models.containers import Container

from exaslct_src.lib.container_log_handler import ContainerLogHandler
from exaslct_src.lib.still_running_logger import StillRunningLogger


class ContainerLogThread(Thread):
    def __init__(self, container: Container, logger, log_file: Path, description: str):
        super().__init__()
        self.complete_log = []
        self.description = description
        self.logger = logger
        self.log_file = log_file
        self.container = container
        self.finish = False
        self.previous_timestamp = None
        self.current_timestamp = None
        self.error_message = None

    def stop(self):
        self.logger.info("Stop ContainerLogThread")
        self.finish = True

    def run(self):
        with ContainerLogHandler(self.log_file, self.logger, self.description) as log_handler:
            still_running_logger = StillRunningLogger(
                self.logger, self.description)
            while not self.finish:
                self.current_timestamp = math.floor(time.time())
                log = self.container.logs(since=self.previous_timestamp, until=self.current_timestamp)
                if len(log) != 0:
                    still_running_logger.log()
                    log_handler.handle_log_line(log)
                log_line = log.decode("utf-8").lower()
                if "error" in log_line \
                        or "exception" in log_line \
                        or "returned with state 1" in log_line:
                    self.logger.info("ContainerLogHandler error message, %s", log_line)
                    self.error_message = log_line
                    self.finish = True
                self.previous_timestamp = self.current_timestamp
                self.complete_log = log_handler.get_complete_log().copy()
                time.sleep(1)
