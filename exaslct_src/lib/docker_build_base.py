import copy
from typing import Dict, Set

from exaslct_src.AbstractMethodException import AbstractMethodException
from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.image_info import ImageInfo, ImageState
from exaslct_src.lib.data.required_task_info import RequiredTaskInfo, RequiredTaskInfoDict
from exaslct_src.lib.docker.docker_analyze_task import DockerAnalyzeImageTask
from exaslct_src.lib.docker.docker_create_image_task import DockerCreateImageTask, DockerCreateImageTaskWithDeps


class DockerBuildBase(DependencyLoggerBaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_build_steps_to_rebuild(self):
        return set(build_config().force_rebuild_from)

    def get_goals(self) -> Set[str]:
        raise AbstractMethodException()

    def get_default_goals(self) -> Set[str]:
        raise AbstractMethodException()

    def get_goal_class_map(self) -> Dict[str, DockerAnalyzeImageTask]:
        raise AbstractMethodException()

    def register_required(self):
        self.analyze_tasks_futures = self.register_dependencies(self.create_analyze_tasks())

    def create_analyze_tasks(self) -> Dict[str, DockerAnalyzeImageTask]:
        goals = self.get_goals()
        self.goal_class_map = self.get_goal_class_map()
        self.available_goals = set(self.goal_class_map.keys())
        self._check_if_build_steps_to_rebuild_are_valid_goals()
        final_goals = self.get_default_goals()
        if len(goals) != 0:
            final_goals = set(goals)
        if final_goals.issubset(self.available_goals):
            tasks = {goal: self.goal_class_map[goal] for goal in final_goals}
            return tasks
        else:
            difference = goals.difference(self.available_goals)
            raise Exception(f"Unknown goal(s) {difference}, "
                            f"following goals are avaialable {self.available_goals}")

    def _check_if_build_steps_to_rebuild_are_valid_goals(self):
        build_steps_to_rebuild = self._get_build_steps_to_rebuild()
        if not build_steps_to_rebuild.issubset(self.available_goals):
            difference = build_steps_to_rebuild.difference(self.available_goals)
            raise Exception(f"Unknown build stages {difference} forced to rebuild, "
                            f"following stages are avaialable {self.available_goals}")

    def create_build_tasks(self, shortcut_build: bool = True) \
            -> Dict[str, DockerCreateImageTask]:
        image_infos = {goal: analyze_task_future.get_output()
                       for goal, analyze_task_future
                       in self.analyze_tasks_futures.items()}
        tasks = self._create_build_tasks_for_image_infos(image_infos, shortcut_build)
        return tasks

    def _create_build_tasks_for_image_infos(self, image_infos: Dict[str, ImageInfo], shortcut_build: bool):
        result = {goal: self._create_build_task_for_image_info(image_info, shortcut_build)
                  for goal, image_info in image_infos.items()}
        return result

    def _create_build_task_for_image_info(
            self, image_info: ImageInfo,
            shortcut_build: bool = True) -> DockerCreateImageTask:
        if self._build_with_depenencies_is_requested(image_info, shortcut_build):
            task_for_image_info = self._create_build_task_with_dependencies(image_info, shortcut_build)
            return task_for_image_info
        else:
            image_name = f"{image_info.target_repository_name}:{image_info.target_tag}"
            task_for_image_info = \
                self.create_child_task(DockerCreateImageTask,
                                       image_name=image_name,
                                       image_info=image_info)
            return task_for_image_info

    def _build_with_depenencies_is_requested(self, image_info: ImageInfo, shortcut_build: bool):
        needs_to_be_build = image_info.image_state == ImageState.NEEDS_TO_BE_BUILD.name
        result = (not shortcut_build or needs_to_be_build) and \
                 len(image_info.depends_on_images) > 0
        return result

    def _create_build_task_with_dependencies(
            self, image_info: ImageInfo,
            shortcut_build: bool = True) -> DockerCreateImageTask:
        required_tasks = \
            self._create_build_tasks_for_image_infos(
                image_info.depends_on_images, shortcut_build)
        required_task_infos = self._create_required_task_infos(required_tasks)
        image_info_copy = copy.copy(image_info)  # TODO looks not nice
        image_info_copy.depends_on_images = {}
        image_name = f"{image_info.target_repository_name}:{image_info.target_tag}"
        task_for_image_info = \
            self.create_child_task(DockerCreateImageTaskWithDeps,
                                   image_name=image_name,
                                   image_info=image_info_copy,
                                   required_task_infos=required_task_infos)
        return task_for_image_info

    def _create_required_task_infos(
            self, required_tasks: Dict[str, DockerCreateImageTask]) -> RequiredTaskInfoDict:
        result = {key: self._create_required_task_info(required_task)
                  for key, required_task in required_tasks.items()}
        return RequiredTaskInfoDict(result)

    def _create_required_task_info(self, required_task: DockerCreateImageTask) -> RequiredTaskInfo:
        required_task_info = RequiredTaskInfo(module_name=required_task.__module__,
                                              class_name=required_task.__class__.__name__,
                                              params=required_task.param_kwargs)
        return required_task_info
