import math
import pathlib
import time
from threading import Thread

from docker.models.containers import Container

from build_utils.lib.still_running_logger import StillRunningLogger
from build_utils.lib.container_log_handler import ContainerLogHandler


class ContainerLogThread(Thread):
    def __init__(self, container: Container, task_id, logger, log_file: pathlib.Path, description:str):
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

    def stop(self):
        self.finish = True

    def my_run(self):
        with ContainerLogHandler(self.log_file, self.logger, self.task_id, self.description) as log_handler:
            still_running_logger = StillRunningLogger(
                self.logger, self.task_id, self.description)
            while not self.finish:
                self.current_timestamp = math.floor(time.time())
                log = self.container.logs(since=self.previous_timestamp, until=self.current_timestamp)
                if len(log) != 0:
                    still_running_logger.log()
                    log_handler.handle_log_line(log)
                self.previous_timestamp = self.current_timestamp
                time.sleep(1)
            self.complete_log = log_handler.get_complete_log()