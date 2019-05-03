import logging
import pathlib
from datetime import datetime
from typing import Generator, List

import luigi
from luigi import LocalTarget

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.still_running_logger import StillRunningLogger, StillRunningLoggerThread
from exaslct_src.lib.task_dependency import TaskDescription, TaskDependency, DependencyType, DependencyState


class StoppingFurtherExecution(Exception):
    pass


class StoppableTask(luigi.Task):
    logger = logging.getLogger('luigi-interface')

    def set_class_targets(self):
        self.failed_target = LocalTarget(build_config().output_directory + "/TASK_FAILED")
        self.create_dependencies_file()
        self.create_timer_files()

    def create_dependencies_file(self):
        self.dependencies_dir = pathlib.Path(build_config().output_directory).joinpath("dependencies")
        self.dependencies_task_dir = self.dependencies_dir.joinpath(self.task_id)
        self.dependencies_task_dir.mkdir(parents=True, exist_ok=True)
        self.dependencies_requires_file = self.dependencies_task_dir.joinpath("requires.json")
        self.dependencies_dynamic_file = self.dependencies_task_dir.joinpath("dynamic.json")

    def create_timer_files(self):
        self.create_timer_base_directories()
        self.create_creation_timer_file()
        self.create_first_run_timer_file()
        self.create_run_timer_file()

    def create_timer_base_directories(self):
        self.timers_dir = pathlib.Path(build_config().output_directory).joinpath("timers")
        self.timers_state_dir = self.timers_dir.joinpath("state")
        self.timers_result_dir = self.timers_dir.joinpath("results")

    def create_creation_timer_file(self):
        self.creation_timer_state_dir = self.timers_state_dir.joinpath("creation")
        self.creation_timer_state_dir.mkdir(parents=True, exist_ok=True)
        self.creation_timer_state_file = self.creation_timer_state_dir.joinpath(self.task_id)

    def create_first_run_timer_file(self):
        self.first_run_timer_state_dir = self.timers_state_dir.joinpath("first_run")
        self.first_run_timer_state_dir.mkdir(parents=True, exist_ok=True)
        self.first_run_timer_state_file = self.first_run_timer_state_dir.joinpath(self.task_id)

    def create_run_timer_file(self):
        self.run_timer_state_dir = self.timers_state_dir.joinpath("run")
        self.run_timer_state_dir.mkdir(parents=True, exist_ok=True)
        self.run_timer_state_file = self.run_timer_state_dir.joinpath(self.task_id)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_class_targets()
        self.write_creation_time()

    def requires(self):
        tasks = self.requires_tasks()
        if tasks is not None:
            if isinstance(tasks, (luigi.Task, list, tuple, dict)):
                return self.handle_requires_value(tasks)
            else:
                return self.handle_requires_generator(tasks)
        else:
            return []

    def handle_requires_value(self, tasks):
        if not self.dependencies_requires_file.exists():
            with self.dependencies_requires_file.open("w") as dependencies_file:
                self.write_dependency(
                    dependencies_file=dependencies_file,
                    dependency_type=DependencyType.requires,
                    dependency_state=DependencyState.requested,
                    index=0, value=tasks)
                return tasks
        else:
            return tasks

    def handle_requires_generator(self, tasks):
        if not self.dependencies_requires_file.exists():
            with self.dependencies_requires_file.open("w") as dependencies_file:
                result = list(self.write_dependencies_for_generator(
                    dependencies_file=dependencies_file,
                    task_generator=tasks,
                    dependency_type=DependencyType.requires))
                return result
        else:
            return tasks

    def requires_tasks(self):
        pass

    def write_creation_time(self):
        if not self.creation_timer_state_file.exists():
            with self.creation_timer_state_file.open("w") as f:
                f.write(str(datetime.now().timestamp()))

    def run(self):
        start_time = datetime.now()
        self.write_first_run_start_time(start_time)
        still_running_logger_thread = self.start_still_running_logger()
        try:
            self.fail_if_any_task_failed()
            with self.dependencies_dynamic_file.open("a") as dependencies_file:
                task_generator = self.run_task()
                if task_generator is not None:
                    yield from self.write_dependencies_for_generator(
                        dependencies_file=dependencies_file,
                        task_generator=task_generator,
                        dependency_type=DependencyType.dynamic)
        finally:
            self.write_run_time(start_time)
            self.stop_still_running_logger(still_running_logger_thread)

    def write_dependencies_for_generator(self, dependencies_file, task_generator,
                                         dependency_type: DependencyType):
        index = 0
        try:
            element = next(task_generator)
            while True:
                self.write_dependency(
                    dependencies_file=dependencies_file,
                    dependency_type=dependency_type,
                    dependency_state=DependencyState.requested,
                    index=index,
                    value=element)
                result = yield element
                element = task_generator.send(result)
                index += 1
        except StopIteration:
            pass

    def write_dependency(self,
                         dependencies_file,
                         dependency_type: DependencyType,
                         dependency_state: DependencyState,
                         index: int,
                         value):
        for task in self.flatten_tasks(value):
            dependency = TaskDependency(source=self.get_task_description(),
                                        target=task.get_task_description(),
                                        type=dependency_type,
                                        index=index,
                                        state=dependency_state)
            dependencies_file.write(f"{dependency.to_json()}")
            dependencies_file.write("\n")

    def get_task_description(self) -> TaskDescription:
        return TaskDescription(id=self.task_id, representation=str(self))

    def flatten_tasks(self, generator: Generator) -> List["StoppableTask"]:
        return [task for task in luigi.task.flatten(generator)
                if isinstance(task, StoppableTask)]

    def run_task(self) -> Generator:
        pass

    def write_first_run_start_time(self, start_time):
        if not self.first_run_timer_state_file.exists():
            with self.first_run_timer_state_file.open("w") as f:
                f.write(str(start_time.timestamp()))

    def start_still_running_logger(self):
        # TODO use larger delay for this StillRunningLogger
        still_running_logger = StillRunningLogger(self.logger, self.task_id, "task")
        still_running_logger_thread = StillRunningLoggerThread(still_running_logger)
        still_running_logger_thread.start()
        return still_running_logger_thread

    def write_run_time(self, start_time):
        timedelta = datetime.now() - start_time
        with self.run_timer_state_file.open("a") as f:
            f.write(str(timedelta.total_seconds()))
            f.write("\n")

    def stop_still_running_logger(self, still_running_logger_thread):
        still_running_logger_thread.stop()
        still_running_logger_thread.join()

    def fail_if_any_task_failed(self):
        if self.failed_target.exists():
            with self.failed_target.open("r") as f:
                failed_task = f.read()
            raise StoppingFurtherExecution("Task %s failed. Stopping further execution." % failed_task)

    def on_success(self):
        now = datetime.now()
        self.log_time_since_first_run(now)
        self.log_time_since_creation(now)
        self.log_time_of_runs()
        super().on_success()

    def log_time_since_creation(self, now):
        if self.creation_timer_state_file.exists():
            with self.creation_timer_state_file.open("r") as f:
                start_time_str = f.read()
            start_time = datetime.fromtimestamp(float(start_time_str))
            timedelta = now - start_time
            self.logger.info("Task %s: Time since creation %s s", self.task_id, timedelta.total_seconds())
            self.timers_result_dir.mkdir(parents=True, exist_ok=True)
            with self.timers_result_dir.joinpath(self.task_id + "_" + "since_creation").open("w") as f:
                f.write(str(timedelta.total_seconds()))

    def log_time_since_first_run(self, now):
        if self.first_run_timer_state_file.exists():
            with self.first_run_timer_state_file.open("r") as f:
                start_time_str = f.read()
            start_time = datetime.fromtimestamp(float(start_time_str))
            timedelta = now - start_time
            self.logger.info("Task %s: Time since first_run %s s", self.task_id, timedelta.total_seconds())
            self.timers_result_dir.mkdir(parents=True, exist_ok=True)
            with self.timers_result_dir.joinpath(self.task_id + "_" + "since_first_run").open("w") as f:
                f.write(str(timedelta.total_seconds()))

    def log_time_of_runs(self):
        if self.run_timer_state_file.exists():
            with self.run_timer_state_file.open("r") as f:
                total_runtime = self.calculate_total_runtime(f.readlines())
            self.logger.info("Task %s: Total runtime of run method %s s", self.task_id, total_runtime)
            with self.timers_result_dir.joinpath(self.task_id + "_" + "total_run").open("w") as f:
                f.write(str(total_runtime))

    def calculate_total_runtime(self, lines):
        total_runtime = 0
        for line in lines:
            seconds_of_run = float(line)
            total_runtime += seconds_of_run
        return total_runtime

    def on_failure(self, exception):
        if not isinstance(exception, StoppingFurtherExecution):
            if not self.failed_target.exists():
                with self.failed_target.open("w") as f:
                    f.write("%s" % self.task_id)
        super().on_failure(exception)

    def __repr__(self):
        """
        Build a task representation like `MyTask(param1=1.5, param2='5')`
        """
        params = self.get_params()
        param_values = self.get_param_values(params, [], self.param_kwargs)

        # Build up task id
        repr_parts = []
        param_objs = dict(params)
        for param_name, param_value in param_values:
            if param_objs[param_name].significant and \
                    param_objs[param_name].visibility == luigi.parameter.ParameterVisibility.PUBLIC:
                repr_parts.append('%s=%s' % (param_name, param_objs[param_name].serialize(param_value)))

        task_str = '{}({})'.format(self.get_task_family(), ', '.join(repr_parts))

        return task_str


class StoppableWrapperTask(StoppableTask):

    def complete(self):
        return all(r.complete() for r in luigi.task.flatten(self.requires()))
