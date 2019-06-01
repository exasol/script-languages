import itertools
from typing import Dict, List

from exaslct_src.lib.stoppable_task import StoppableTask
from exaslct_src.lib.data.required_task_info import RequiredTaskInfo
from exaslct_src.lib.docker.docker_create_image_task import DockerCreateImageTask, DockerCreateImageTaskWithDeps


class TaskCreatorFromBuildTasks:

    def create_tasks_for_flavors(
            self, build_tasks_for_flavors: Dict[str, Dict[str, DockerCreateImageTask]]) \
            -> List[StoppableTask]:
        tasks_per_flavor = [self.create_tasks_for_build_tasks(build_task)
                            for flavor_path, build_task in build_tasks_for_flavors.items()]
        return list(itertools.chain.from_iterable(tasks_per_flavor))

    def create_tasks_for_build_tasks(self, build_tasks: Dict[str, DockerCreateImageTask]) \
            -> List[StoppableTask]:
        tasks_per_goal = [self.create_tasks_for_build_task(build_task)
                          for goal, build_task in build_tasks.items()]
        return list(itertools.chain.from_iterable(tasks_per_goal))

    def create_tasks_for_build_task(self, build_task: DockerCreateImageTask) \
            -> List[StoppableTask]:
        if isinstance(build_task, DockerCreateImageTaskWithDeps):
            tasks = self.create_tasks_for_build_tasks(build_task.requires_tasks())
            task = self.create_task(build_task)
            return [task] + tasks
        else:
            task = self.create_task(build_task)
            return [task]

    def create_task(self, build_task):
        required_task_info = self.create_required_task_info(build_task)
        task = self.create_task_with_required_tasks(build_task, required_task_info)
        return task

    def create_required_task_info(self, build_task: DockerCreateImageTask):
        required_task_info = \
            RequiredTaskInfo(module_name=build_task.__module__,
                             class_name=build_task.__class__.__name__,
                             params=build_task.param_kwargs)
        return required_task_info

    def create_task_with_required_tasks(self, build_task, required_task_info) -> StoppableTask:
        pass
