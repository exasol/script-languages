from pathlib import Path

from exaslct_src.lib.base.timeable_base_task import TimeableBaseTask
from exaslct_src.lib.stoppable_task import StoppingFurtherExecution


class StoppableBaseTask(TimeableBaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed_target = Path(super()._get_output_path_for_job(),"TASKED_FAILED")

    def run(self):
        self.fail_if_any_task_failed()
        task_generator = super().run()
        if task_generator is not None:
            yield from task_generator

    def fail_if_any_task_failed(self):
        if self.failed_target.exists():
            with self.failed_target.open("r") as f:
                failed_task = f.read()
            raise StoppingFurtherExecution("Task %s failed. Stopping further execution." % failed_task)

    def on_failure(self, exception):
        if not isinstance(exception, StoppingFurtherExecution):
            if not self.failed_target.exists():
                with self.failed_target.open("w") as f:
                    f.write("%s" % self.task_id)
        super().on_failure(exception)
