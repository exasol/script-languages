import luigi
from luigi import LocalTarget


class StoppingFurtherExecution(Exception):
    pass


class StoppableTask(luigi.Task):
    failed_target = LocalTarget("FAILED")

    def run(self):
        self.check_if_any_task_failed()
        result = self.my_run()
        if result is not None:
            yield from result

    def check_if_any_task_failed(self):
        if self.failed_target.exists():
            with self.failed_target.open("r") as f:
                failed_task = f.read()
            raise StoppingFurtherExecution("Task %s failed. Stopping further execution." % failed_task)

    def my_run(self):
        pass

    def on_failure(self, exception):
        if not isinstance(exception, StoppingFurtherExecution) and not self.failed_target.exists():
            with self.failed_target.open("w") as f:
                f.write("%s" % self.task_id)
        super().on_failure(exception)
