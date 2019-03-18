from enum import Enum

import luigi

from build_utils.release_container_task import ReleaseContainerTask
from build_utils.docker_build import DockerBuild_Release, DockerBuild_BaseTestBuildRun, DockerBuild_FlavorTestBuildRun
from build_utils.flavor_task import FlavorWrapperTask


class ReleaseContainer_Release(ReleaseContainerTask):
    def get_release_task(self, flavor_path):
        return DockerBuild_Release(flavor_path)


class ReleaseContainer_BaseTest(ReleaseContainerTask):
    def get_release_task(self, flavor_path):
        return DockerBuild_BaseTestBuildRun(flavor_path)


class ReleaseContainer_FlavorTest(ReleaseContainerTask):
    def get_release_task(self, flavor_path):
        return DockerBuild_FlavorTestBuildRun(flavor_path)


class ReleaseType(Enum):
    Release = 1,
    BaseTest = 2,
    FlavorTest = 3


class ReleaseContainer(FlavorWrapperTask):
    release_types = luigi.ListParameter(["Release"])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actul_release_types = [ReleaseType[release_type] for release_type in self.release_types]

    def requires(self):
        return [self.generate_tasks_for_flavor(flavor_path) for flavor_path in self.actual_flavor_paths]

    def generate_tasks_for_flavor(self, flavor_path):
        result = []
        if ReleaseType.Release in self.actul_release_types:
            result.append(ReleaseContainer_Release(flavor_path=flavor_path))
        if ReleaseType.BaseTest in self.actul_release_types:
            result.append(ReleaseContainer_BaseTest(flavor_path=flavor_path))
        if ReleaseType.FlavorTest in self.actul_release_types:
            result.append(ReleaseContainer_FlavorTest(flavor_path=flavor_path))
        return result
