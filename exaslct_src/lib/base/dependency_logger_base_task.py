from pathlib import Path
from typing import List, Generator

import luigi
from luigi import Task

from exaslct_src.lib.base.base_task import BaseTask
from exaslct_src.lib.base.stoppable_base_task import StoppableBaseTask
from exaslct_src.lib.task_dependency import TaskDescription, TaskDependency, DependencyType, DependencyState


class DependencyLoggerBaseTask(StoppableBaseTask):

    def _get_dependencies_path_for_job(self):
        return Path(super()._get_output_path_for_job(), "dependencies")

    def _get_dependencies_path(self):
        path = Path(self._get_dependencies_path_for_job(), self.task_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_dependencies_requires_path(self):
        return Path(self._get_dependencies_path(), "requires")

    def _get_dependencies_dynamic_path(self):
        return Path(self._get_dependencies_path(), "dynamic")

    def handle_requires_value(self, tasks):
        dependency_path = self._get_dependencies_requires_path()
        if not dependency_path.exists():
            with dependency_path.open("w") as dependencies_file:
                self.write_dependency(
                    dependencies_file=dependencies_file,
                    dependency_type=DependencyType.requires,
                    dependency_state=DependencyState.requested,
                    index=0, value=tasks)
                return tasks
        else:
            return tasks

    def handle_requires_generator(self, tasks):
        dependency_path = self._get_dependencies_requires_path()
        if not dependency_path.exists():
            with dependency_path.open("w") as dependencies_file:
                result = list(self.write_dependencies_for_generator(
                    dependencies_file=dependencies_file,
                    task_generator=tasks,
                    dependency_type=DependencyType.requires))
                return result
        else:
            return tasks

    def write_dependencies_for_generator(self,
                                         dependencies_file,
                                         task_generator,
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

    def requires(self):
        tasks = super().requires()
        if tasks is not None:
            if isinstance(tasks, (Task, list, tuple, dict)):
                return self.handle_requires_value(tasks)
            else:
                return self.handle_requires_generator(tasks)
        else:
            return []

    def run(self):
        dependency_path = self._get_dependencies_dynamic_path()
        with dependency_path.open("a") as dependencies_file:
            task_generator = super().run()
            if task_generator is not None:
                yield from self.write_dependencies_for_generator(
                    dependencies_file=dependencies_file,
                    task_generator=task_generator,
                    dependency_type=DependencyType.dynamic)

    def get_task_description(self) -> TaskDescription:
        return TaskDescription(id=self.task_id, representation=str(self))

    def flatten_tasks(self, generator: Generator) -> List[BaseTask]:
        return [task for task in luigi.task.flatten(generator)
                if isinstance(task, BaseTask)]
