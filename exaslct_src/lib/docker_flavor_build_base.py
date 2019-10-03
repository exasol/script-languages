from typing import Dict, Set

from exaslct_src.lib.docker.docker_analyze_task import DockerAnalyzeImageTask
from exaslct_src.lib.docker.docker_flavor_image_task import DockerFlavorAnalyzeImageTask
from exaslct_src.lib.docker_build_base import DockerBuildBase
from exaslct_src.lib.flavor_task import FlavorBaseTask, FlavorsBaseTask


class DockerFlavorBuildBase(FlavorBaseTask, DockerBuildBase):

    # TODO order pull for images which share dependencies

    def get_goal_class_map(self) -> Dict[str, DockerAnalyzeImageTask]:
        module_name_for_build_steps = self.flavor_path.replace("/", "_").replace(".", "_")
        available_tasks = [self.create_child_task_with_common_params(subclass)
                           for subclass
                           in DockerFlavorAnalyzeImageTask.__subclasses__()
                           if subclass.__module__ == module_name_for_build_steps]
        goal_class_map = {task.get_build_step(): task for task in available_tasks}
        return goal_class_map

    def get_default_goals(self) -> Set[str]:
        goals = {"release", "base_test_build_run", "flavor_test_build_run"}
        return goals
