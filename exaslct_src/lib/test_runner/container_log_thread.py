import math
import pathlib
import time
from threading import Thread

from docker.models.containers import Container

from exaslct_src.lib.still_running_logger import StillRunningLogger
from exaslct_src.lib.container_log_handler import ContainerLogHandler

class ContainerLogThread(Thread):
    def __init__(self, container: Container, task_id, logger, log_file: pathlib.Path, description: str):
        super().__init__()
        self.complete_log = []
        self.description = description
        self.logger = logger
        self.task_id = task_id
        self.log_file = log_file
        self.container = container
        self.finish = False
        self.previous_timestamp = math.floor(time.time())
        self.current_timestamp = self.previous_timestamp
        self.error_message = None

    def stop(self):
        self.finish = True

    def run(self):
        with ContainerLogHandler(self.log_file, self.logger, self.task_id, self.description) as log_handler:
            still_running_logger = StillRunningLogger(
                self.logger, self.task_id, self.description)
            while not self.finish:
                self.current_timestamp = math.floor(time.time())
                log = self.container.logs(since=self.previous_timestamp, until=self.current_timestamp)
                if len(log) != 0:
                    still_running_logger.log()
                    log_handler.handle_log_line(log)
                log_line = log.decode("utf-8").lower()
                if "error" in log_line or "exception" in log_line or "returned with state 1" in log_line:
                    self.error_message = log_line
                    self.finish = True
                self.previous_timestamp = self.current_timestamp
                time.sleep(1)
            self.complete_log = log_handler.get_complete_log()
