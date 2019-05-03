from typing import Dict, Set

from exaslct_src.lib.docker.docker_analyze_task import DockerAnalyzeImageTask
from exaslct_src.lib.docker.docker_flavor_image_task import DockerFlavorAnalyzeImageTask
from exaslct_src.lib.docker_build_base import DockerBuildBase
from exaslct_src.lib.flavor_task import FlavorTask


class DockerFlavorBuildBase(FlavorTask,DockerBuildBase):

    # TODO order pull for images which share dependencies
    # TODO add retag option, pull from one repository-name but build with another one

    def get_goal_class_map(self, params)->Dict[str,DockerAnalyzeImageTask]:
        # needed to detect the build_steps
        import exaslct_src.build_steps
        available_tasks = [subclass(**params)
                           for subclass in DockerFlavorAnalyzeImageTask.__subclasses__()]
        goal_class_map = {task.get_build_step(): task for task in available_tasks}
        return goal_class_map

    def get_default_goals(self)->Set[str]:
        goals = {"release", "base_test_build_run", "flavor_test_build_run"}
        return goals