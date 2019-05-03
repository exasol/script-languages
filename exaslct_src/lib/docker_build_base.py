import copy
from abc import abstractmethod
from typing import Dict, Set

from luigi import LocalTarget

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from exaslct_src.lib.data.image_info import ImageInfo, ImageState
from exaslct_src.lib.data.required_task_info import RequiredTaskInfo
from exaslct_src.lib.docker.docker_create_image_task import DockerCreateImageTask, DockerCreateImageTaskWithDeps
from exaslct_src.lib.docker.docker_analyze_task import DockerAnalyzeImageTask
from exaslct_src.lib.stoppable_task import StoppableTask

# TODO abstract flavor_path away
class DockerBuildBase(StoppableTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._goals = self.get_goals()
        self.build_steps_to_rebuild = set(build_config().force_rebuild_from)

    def get_goals(self):
        pass

    def requires_tasks(self) -> Dict[str, Dict[str, DockerAnalyzeImageTask]]:
        return {flavor_path: self.generate_tasks_of_goals_for_flavor(flavor_path)
                for flavor_path in self.actual_flavor_paths}

    def generate_tasks_of_goals_for_flavor(self, flavor_path) -> Dict[set, DockerAnalyzeImageTask]:
        self.goal_class_map = self.get_goal_class_map({"flavor_path":flavor_path})
        self.available_goals = set(self.goal_class_map.keys())
        self.check_if_build_steps_to_rebuild_are_valid_goals()
        goals = self.get_default_goals()
        if len(self._goals) != 0:
            goals = set(self._goals)
        if goals.issubset(self.available_goals):
            tasks = {goal: self.goal_class_map[goal] for goal in goals}
            return tasks
        else:
            difference = goals.difference(self.available_goals)
            raise Exception(f"Unknown goal(s) {difference}, "
                            f"following goals are avaialable {self.available_goals}")

    @abstractmethod
    def get_default_goals(self)->Set[str]:
        pass

    @abstractmethod
    def get_goal_class_map(self, params)->Dict[str,DockerAnalyzeImageTask]:
        pass

    def check_if_build_steps_to_rebuild_are_valid_goals(self):
        if not self.build_steps_to_rebuild.issubset(self.available_goals):
            difference = self.build_steps_to_rebuild.difference(self.available_goals)
            raise Exception(f"Unknown build stages {difference} forced to rebuild, "
                            f"following stages are avaialable {self.available_goals}")

    def create_build_tasks_for_all_flavors(self, shortcut_build: bool = True) \
            -> Dict[str, Dict[str, DockerCreateImageTask]]:
        result = {
            flavor_path:
                self.create_build_tasks_for_image_info_targets(
                    image_info_targets, shortcut_build)
            for flavor_path, image_info_targets in self.input().items()
        }
        return result

    def create_build_tasks_for_image_infos(
            self, image_infos: Dict[str, ImageInfo],
            shortcut_build: bool = True) \
            -> Dict[str, DockerCreateImageTask]:
        result = {key: self.create_build_task_for_image_info(image_info, shortcut_build)
                  for key, image_info in image_infos.items()}
        return result

    def create_build_tasks_for_image_info_targets(
            self, image_info_targets: Dict[str, Dict[str, LocalTarget]],
            shortcut_build: bool = True) \
            -> Dict[str, DockerCreateImageTask]:
        image_infos = DependencyImageInfoCollector().get_from_dict_of_inputs(image_info_targets)
        return self.create_build_tasks_for_image_infos(image_infos, shortcut_build)

    def create_build_task_for_image_info(
            self, image_info: ImageInfo,
            shortcut_build: bool = True) -> DockerCreateImageTask:
        if (self.build_requested(image_info, shortcut_build)):
            task_for_image_info = self.create_build_task_with_dependencies(image_info, shortcut_build)
            return task_for_image_info
        else:
            task_for_image_info = \
                DockerCreateImageTask(
                    image_name=f"{image_info.name}:{image_info.tag}",
                    image_info_json=image_info.to_json(indent=None))
            return task_for_image_info

    def build_requested(self, image_info: ImageInfo, shortcut_build: bool):
        needs_to_be_build = image_info.image_state == ImageState.NEEDS_TO_BE_BUILD.name
        result = (not shortcut_build or needs_to_be_build) and \
                 len(image_info.depends_on_images) > 0
        return result

    def create_build_task_with_dependencies(
            self, image_info: ImageInfo,
            shortcut_build: bool = True) -> DockerCreateImageTask:
        required_tasks = \
            self.create_build_tasks_for_image_infos(
                image_info.depends_on_images, shortcut_build)
        required_task_infos_json = {
            goal: required_task.to_json(indent=None)
            for goal, required_task
            in self.create_required_task_infos(required_tasks).items()
        }
        image_info_copy = copy.copy(image_info)
        image_info_copy.depends_on_images = {}
        task_for_image_info = \
            DockerCreateImageTaskWithDeps(
                image_name=f"{image_info.name}:{image_info.tag}",
                image_info_json=image_info_copy.to_json(indent=None),
                required_task_infos_json=required_task_infos_json)
        return task_for_image_info

    def create_required_task_infos(
            self, required_tasks: Dict[str, DockerCreateImageTask]) -> Dict[str, RequiredTaskInfo]:
        result = {key: self.create_required_task_info(required_task)
                  for key, required_task in required_tasks.items()}
        return result

    def create_required_task_info(self, required_task: DockerCreateImageTask) -> RequiredTaskInfo:
        required_task_info = RequiredTaskInfo(module_name=required_task.__module__,
                                              class_name=required_task.__class__.__name__,
                                              params=required_task.param_kwargs)
        return required_task_info

