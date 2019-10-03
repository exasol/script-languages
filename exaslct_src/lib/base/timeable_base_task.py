from datetime import datetime
from pathlib import Path

from exaslct_src.lib.base.base_task import BaseTask


class TimeableBaseTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_creation_time()

    def _get_timers_path_for_job(self):
        return Path(super()._get_output_path_for_job(), "timers")

    def _get_timers_path(self):
        return Path(self._get_timers_path_for_job(), self.task_id)

    def _get_timers_state_path(self):
        path = Path(self._get_timers_path(), "state")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_timers_result_path(self):
        path = Path(self._get_timers_path(), "result")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_timers_creation_state_path(self):
        return Path(self._get_timers_state_path(), "creation")

    def _get_timers_first_run_state_path(self):
        return Path(self._get_timers_state_path(), "first_run")

    def _get_timers_runs_state_path(self):
        return Path(self._get_timers_state_path(), "runs")

    def _get_timers_creation_result_path(self):
        return Path(self._get_timers_result_path(), "time_since_creation")

    def _get_timers_first_run_result_path(self):
        return Path(self._get_timers_result_path(), "time_since_first_run")

    def _get_timers_runs_result_path(self):
        return Path(self._get_timers_result_path(), "total_time_of_runs")

    def write_creation_time(self):
        state_path = self._get_timers_creation_state_path()
        if not state_path.exists():
            with state_path.open("w") as f:
                f.write(str(datetime.now().timestamp()))

    def write_first_run_start_time(self, start_time):
        state_path = self._get_timers_first_run_state_path()
        if not state_path.exists():
            with state_path.open("w") as f:
                f.write(str(start_time.timestamp()))

    def write_runs_time(self, start_time):
        state_path = self._get_timers_runs_state_path()
        timedelta = datetime.now() - start_time
        with state_path.open("a") as f:
            f.write(str(timedelta.total_seconds()))
            f.write("\n")

    def on_success(self):
        super().on_success()
        now = datetime.now()
        self.log_time_since_first_run(now)
        self.log_time_since_creation(now)
        self.log_time_of_runs()

    def log_time_since_creation(self, now):
        state_path = self._get_timers_creation_state_path()
        if state_path.exists():
            with state_path.open("r") as f:
                start_time_str = f.read()
            start_time = datetime.fromtimestamp(float(start_time_str))
            timedelta = now - start_time
            self.logger.info(f"Time since creation {timedelta.total_seconds()} s")
            with self._get_timers_creation_result_path().open("w") as f:
                f.write(str(timedelta.total_seconds()))
                f.write("\n")

    def log_time_since_first_run(self, now):
        state_path = self._get_timers_first_run_state_path()
        if state_path.exists():
            with state_path.open("r") as f:
                start_time_str = f.read()
            start_time = datetime.fromtimestamp(float(start_time_str))
            timedelta = now - start_time
            self.logger.info("Time since first_run %s s", timedelta.total_seconds())
            with self._get_timers_first_run_result_path().open("w") as f:
                f.write(str(timedelta.total_seconds()))
                f.write("\n")

    def log_time_of_runs(self):
        state_path = self._get_timers_runs_state_path()
        if state_path.exists():
            with state_path.open("r") as f:
                total_runtime = self.calculate_total_runtime(f.readlines())
            self.logger.info("Total runtime of run method %s s", total_runtime)
            with self._get_timers_runs_result_path().open("w") as f:
                f.write(str(total_runtime))
                f.write("\n")

    def calculate_total_runtime(self, lines):
        total_runtime = 0
        for line in lines:
            seconds_of_run = float(line)
            total_runtime += seconds_of_run
        return total_runtime

    def run(self):
        start_time = datetime.now()
        self.write_first_run_start_time(start_time)
        try:
            task_generator = super().run()
            if task_generator is not None:
                yield from task_generator
        finally:
            self.write_runs_time(start_time)
