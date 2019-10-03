import threading
import time
from datetime import datetime, timedelta

from exaslct_src.lib.log_config import log_config



class StillRunningLoggerThread(threading.Thread):
    def __init__(self, still_running_logger: "StillRunningLogger"):
        super().__init__()
        self.still_running_logger = still_running_logger
        self.finish = False

    def stop(self):
        self.finish = True

    def run(self):
        while not self.finish:
            self.still_running_logger.log()
            time.sleep(1)

class StillRunningLogger():

    def __init__(self, logger, description):
        self._description = description
        self._logger = logger
        self._log_config = log_config()
        self._previous_time = datetime.now()

    def log(self, message=None):

        timedelta_between = timedelta(seconds=self._log_config.seconds_between_still_running_logs)
        if self._previous_time + timedelta_between <= datetime.now():
            if message is None:
                self._logger.info("Still running %s.", self._description)
            else:
                self._logger.info("Still running %s. Message: %s", self._description, message)
            self._previous_time = datetime.now()
