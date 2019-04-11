import logging
from datetime import datetime

import luigi
from luigi import LocalTarget

from build_utils.lib.build_config import build_config


class StoppingFurtherExecution(Exception):
    pass


class StoppableTask(luigi.Task):
    logger = logging.getLogger('luigi-interface')

    failed_target = LocalTarget(build_config().output_directory + "/TASK_FAILED")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        start_time = datetime.now()
        self.fail_if_any_task_failed()
        result = self.run_task()
        if result is not None:
            yield from result
        timedelta = datetime.now() - start_time
        self.logger.info("Task %s: Self-time was %s s", self.task_id, timedelta.total_seconds())

    def fail_if_any_task_failed(self):
        if self.failed_target.exists():
            with self.failed_target.open("r") as f:
                failed_task = f.read()
            raise StoppingFurtherExecution("Task %s failed. Stopping further execution." % failed_task)

    def run_task(self):
        pass

    def on_failure(self, exception):
        if not isinstance(exception, StoppingFurtherExecution):
            if not self.failed_target.exists():
                with self.failed_target.open("w") as f:
                    f.write("%s" % self.task_id)
        super().on_failure(exception)
