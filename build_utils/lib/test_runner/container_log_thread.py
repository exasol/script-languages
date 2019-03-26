import math
import pathlib
import time
from threading import Thread

from docker.models.containers import Container


class ContainerLogThread(Thread):
    def __init__(self, container: Container, target: pathlib.Path):
        super().__init__()
        self.target = target
        self.container = container
        self.finish = False
        self.previous_timestamp = math.floor(time.time())
        self.current_timestamp = self.previous_timestamp

    def stop(self):
        self.finish = True

    def run(self):
        with self.target.open("w") as file:
            while not self.finish:
                self.current_timestamp = math.floor(time.time())
                log = self.container.logs(since=self.previous_timestamp, until=self.current_timestamp)
                if len(log) != 0:
                    file.write(log.decode("utf-8"))
                self.previous_timestamp = self.current_timestamp
                time.sleep(1)