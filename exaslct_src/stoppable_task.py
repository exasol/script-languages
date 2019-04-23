import logging
import pathlib
from datetime import datetime, timedelta

import luigi
from luigi import LocalTarget

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.still_running_logger import StillRunningLogger, StillRunningLoggerThread


class StoppingFurtherExecution(Exception):
    pass


class StoppableTask(luigi.Task):
    logger = logging.getLogger('luigi-interface')

    failed_target = LocalTarget(build_config().output_directory + "/TASK_FAILED")

    timers_dir = pathlib.Path(build_config().output_directory).joinpath("timers")
    timers_state_dir = timers_dir.joinpath("state")
    timers_result_dir = timers_dir.joinpath("results")

    creation_timer_state_dir = timers_state_dir.joinpath("creation")
    first_run_timer_state_dir = timers_state_dir.joinpath("first_run")
    run_timer_state_dir = timers_state_dir.joinpath("run")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.creation_timer_state_dir.mkdir(parents=True, exist_ok=True)
        self.creation_timer_state_file = self.creation_timer_state_dir.joinpath(self.task_id)
        if not self.creation_timer_state_file.exists():
            with self.creation_timer_state_file.open("w") as f:
                f.write(str(datetime.now().timestamp()))
        self.run_timer_state_dir.mkdir(parents=True, exist_ok=True)
        self.run_timer_state_file = self.run_timer_state_dir.joinpath(self.task_id)
        self.first_run_timer_state_dir.mkdir(parents=True, exist_ok=True)
        self.first_run_timer_state_file = self.first_run_timer_state_dir.joinpath(self.task_id)

    def run(self):
        start_time = datetime.now()
        if not self.first_run_timer_state_file.exists():
            with self.first_run_timer_state_file.open("w") as f:
                f.write(str(start_time.timestamp()))
        still_running_logger = StillRunningLogger(self.logger, self.task_id, "task")
        thread = StillRunningLoggerThread(still_running_logger)
        thread.start()
        try:
            self.fail_if_any_task_failed()
            result = self.run_task()
            if result is not None:
                yield from result
        finally:
            timedelta = datetime.now() - start_time
            with self.run_timer_state_file.open("a") as f:
                f.write(str(timedelta.total_seconds()))
                f.write("\n")
            thread.stop()
            thread.join()

    def fail_if_any_task_failed(self):
        if self.failed_target.exists():
            with self.failed_target.open("r") as f:
                failed_task = f.read()
            raise StoppingFurtherExecution("Task %s failed. Stopping further execution." % failed_task)

    def run_task(self):
        pass

    def on_success(self):
        now = datetime.now()
        self.log_time_since_first_run(now)
        self.log_time_since_creation(now)
        self.log_time_of_runs()
        super().on_success()

    def log_time_since_creation(self,now):
        if self.creation_timer_state_file.exists():
            with self.creation_timer_state_file.open("r") as f:
                start_time_str = f.read()
            start_time = datetime.fromtimestamp(float(start_time_str))
            timedelta =  now - start_time
            self.logger.info("Task %s: Time since creation %s s", self.task_id, timedelta.total_seconds())
            self.timers_result_dir.mkdir(parents=True, exist_ok=True)
            with self.timers_result_dir.joinpath(self.task_id+"_"+"since_creation").open("w") as f:
                f.write(str(timedelta.total_seconds()))

    def log_time_since_first_run(self,now):
        if self.first_run_timer_state_file.exists():
            with self.first_run_timer_state_file.open("r") as f:
                start_time_str = f.read()
            start_time = datetime.fromtimestamp(float(start_time_str))
            timedelta = now - start_time
            self.logger.info("Task %s: Time since first_run %s s", self.task_id, timedelta.total_seconds())
            self.timers_result_dir.mkdir(parents=True, exist_ok=True)
            with self.timers_result_dir.joinpath(self.task_id+"_"+"since_first_run").open("w") as f:
                f.write(str(timedelta.total_seconds()))

    def log_time_of_runs(self):
        if self.run_timer_state_file.exists():
            with self.run_timer_state_file.open("r") as f:
                total_runtime = 0
                for line in f.readlines():
                    seconds_of_run = float(line)
                    total_runtime += seconds_of_run
            self.logger.info("Task %s: Total runtime of run method %s s", self.task_id, total_runtime)
            with self.timers_result_dir.joinpath(self.task_id+"_"+"total_run").open("w") as f:
                f.write(str(total_runtime))

    def on_failure(self, exception):
        if not isinstance(exception, StoppingFurtherExecution):
            if not self.failed_target.exists():
                with self.failed_target.open("w") as f:
                    f.write("%s" % self.task_id)
        super().on_failure(exception)
