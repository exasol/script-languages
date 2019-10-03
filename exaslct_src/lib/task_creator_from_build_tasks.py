import itertools
from typing import Dict, List

from exaslct_src.AbstractMethodException import AbstractMethodException
from exaslct_src.lib.base.base_task import BaseTask

from exaslct_src.lib.data.required_task_info import RequiredTaskInfo
from exaslct_src.lib.docker.docker_create_image_task import DockerCreateImageTask, DockerCreateImageTaskWithDeps


class TaskCreatorFromBuildTasks:

    def create_tasks_for_build_tasks(self, build_tasks: Dict[str, DockerCreateImageTask]) \
            -> List[BaseTask]:
        tasks_per_goal = [self._create_tasks_for_build_task(build_task)
                          for goal, build_task in build_tasks.items()]
        return list(itertools.chain.from_iterable(tasks_per_goal))

    def _create_tasks_for_build_task(self, build_task: DockerCreateImageTask) \
            -> List[BaseTask]:
        if isinstance(build_task, DockerCreateImageTaskWithDeps):
            tasks = self.create_tasks_for_build_tasks(build_task.required_tasks)
            task = self._create_task(build_task)
            return [task] + tasks
        else:
            task = self._create_task(build_task)
            return [task]

    def _create_task(self, build_task):
        required_task_info = self._create_required_task_info(build_task)
        task = self.create_task_with_required_tasks(build_task, required_task_info)
        return task

    def _create_required_task_info(self, build_task: DockerCreateImageTask):
        required_task_info = \
            RequiredTaskInfo(module_name=build_task.__module__,
                             class_name=build_task.__class__.__name__,
                             params=build_task.param_kwargs)
        return required_task_info

    def create_task_with_required_tasks(self, build_task, required_task_info) -> BaseTask:
        raise AbstractMethodException()
