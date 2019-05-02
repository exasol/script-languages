from collections import deque
from typing import Dict, Set, List

import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.docker.docker_pull_or_build_flavor_image_task import DockerPullOrBuildFlavorImageTask
from exaslct_src.lib.docker.docker_pull_or_build_image_tasks import DockerPullOrBuildImageTask
from exaslct_src.lib.flavor_task import FlavorWrapperTask


class DockerBuild_UDFClientDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "udfclient_deps"

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"01_nodoc": "ext/01_nodoc"}

    def get_path_in_flavor(self):
        return "flavor_base"


class DockerBuild_LanguageDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "language_deps"

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"scripts": "ext/scripts"}

    def requires_tasks(self):
        return {"udfclient_deps": DockerBuild_UDFClientDeps(flavor_path=self.flavor_path)}

    def get_path_in_flavor(self):
        return "flavor_base"


class DockerBuild_BuildDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "build_deps"

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"01_nodoc": "ext/01_nodoc", "scripts": "ext/scripts"}

    def get_path_in_flavor(self):
        return "flavor_base"


class DockerBuild_BuildRun(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "build_run"

    def requires_tasks(self):
        return {"build_deps": DockerBuild_BuildDeps(flavor_path=self.flavor_path),
                "language_deps": DockerBuild_LanguageDeps(flavor_path=self.flavor_path)}

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"src": "src"}

    def get_path_in_flavor(self):
        return "flavor_base"


class DockerBuild_BaseTestDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "base_test_deps"

    def requires_tasks(self):
        return {"build_deps": DockerBuild_BuildDeps(flavor_path=self.flavor_path)}

    def get_path_in_flavor(self):
        return "flavor_base"


class DockerBuild_BaseTestBuildRun(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "base_test_build_run"

    def requires_tasks(self):
        return {"base_test_deps": DockerBuild_BaseTestDeps(flavor_path=self.flavor_path),
                "language_deps": DockerBuild_LanguageDeps(flavor_path=self.flavor_path)}

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"src": "src", "emulator": "emulator"}

    def get_path_in_flavor(self):
        return "flavor_base"


class DockerBuild_FlavorBaseDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "flavor_base_deps"

    def requires_tasks(self):
        return {"language_deps": DockerBuild_LanguageDeps(flavor_path=self.flavor_path)}

    def get_additional_build_directories_mapping(self):
        return {"01_nodoc": "ext/01_nodoc", "scripts": "ext/scripts"}

    def get_path_in_flavor(self):
        return "flavor_base"


class DockerBuild_FlavorCustomization(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "flavor_customization"

    def requires_tasks(self):
        return {"flavor_base_deps": DockerBuild_FlavorBaseDeps(flavor_path=self.flavor_path)}


class DockerBuild_FlavorTestBuildRun(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "flavor_test_build_run"

    def requires_tasks(self):
        return {"flavor_customization": DockerBuild_FlavorCustomization(flavor_path=self.flavor_path),
                "base_test_build_run": DockerBuild_BaseTestBuildRun(flavor_path=self.flavor_path)}

    def get_path_in_flavor(self):
        return "flavor_base"


class DockerBuild_Release(DockerPullOrBuildFlavorImageTask):
    def get_build_step(self) -> str:
        return "release"

    def requires_tasks(self):
        return {"flavor_customization": DockerBuild_FlavorCustomization(flavor_path=self.flavor_path),
                "build_run": DockerBuild_BuildRun(flavor_path=self.flavor_path),
                "language_deps": DockerBuild_LanguageDeps(flavor_path=self.flavor_path)}

    def get_path_in_flavor(self):
        return "flavor_base"


# TODO optimize build time, by only pulling absolut necassry images,
#       for example, if release container is on docker hub,
#       we only need to pull this one
#           - first compute for all build steps the hash
#           - check top down which images are available in local or docker hub cache
#           - only pull absolut necassry images
# TODO add retag option, pull from one repository-name but build with another one
# TODO add option release type
# TODO allow partial builds up to certain build step to split the build into multiple stage in travis
class DockerBuild(FlavorWrapperTask):
    goals = luigi.ListParameter([])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self.goal_class_map = {
            "udfclient_deps": DockerBuild_UDFClientDeps,
            "language_deps": DockerBuild_LanguageDeps,
            "build_deps": DockerBuild_BuildDeps,
            "build_run": DockerBuild_BuildRun,
            "flavor_base_deps": DockerBuild_FlavorBaseDeps,
            "flavor_customization": DockerBuild_FlavorCustomization,
            "release": DockerBuild_Release,
            "base_test_deps": DockerBuild_BaseTestDeps,
            "base_test_build_run": DockerBuild_BaseTestBuildRun,
            "flavor_test_build_run": DockerBuild_FlavorTestBuildRun
        }
        self.available_goals = set(self.goal_class_map.keys())
        build_stages_to_rebuild = set(self._build_config.force_rebuild_from)
        if not build_stages_to_rebuild.issubset(
                self.available_goals):
            difference = build_stages_to_rebuild.difference(self.available_goals)
            raise Exception(f"Unknown build stages {difference} forced to rebuild, "
                            f"following stages are avaialable {self.available_goals}")

    def requires_tasks(self) -> List[Set[DockerPullOrBuildImageTask]]:
        return [self.generate_tasks_of_goals_for_flavor(flavor_path)
                for flavor_path in self.actual_flavor_paths]

    def generate_dependencies_for_build_tasks(self):
        return [self.generate_dependencies_for_build_tasks_of_flavor(flavor_path)
                for flavor_path in self.actual_flavor_paths]

    def generate_dependencies_for_build_tasks_of_flavor(self, flavor_path):
        tasks = self.generate_tasks_of_goals_for_flavor(flavor_path)
        dependencies = self.get_dependencies(tasks)
        return dependencies

    def generate_tasks_of_goals_for_flavor(self, flavor_path) -> Set[DockerPullOrBuildImageTask]:
        goals = {"release", "base_test_build_run", "flavor_test_build_run"}
        if len(self.goals) != 0:
            goals = set(self.goals)
        if goals.issubset(self.available_goals):
            tasks = {self.goal_class_map[goal](flavor_path=flavor_path) for goal in goals}
            return tasks
        else:
            difference = goals.difference(self.available_goals)
            raise Exception(f"Unknown goal(s) {difference}, "
                            f"following goals are avaialable {self.available_goals}")

    def get_dependencies(self, tasks) -> Set[DockerPullOrBuildImageTask]:
        dependencies = list(tasks)
        task_deque = deque(tasks)
        while len(task_deque) != 0:
            current_task = task_deque.pop()
            requirements = [task for task
                            in luigi.task.flatten(current_task.requires())
                            if isinstance(task, DockerPullOrBuildImageTask)]
            dependencies.extend(requirements)
            task_deque.extend(requirements)
        dependencies = set(dependencies)
        return dependencies
