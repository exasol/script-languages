import luigi

from exaslct_src.docker_build import DockerBuild_Release, DockerBuild_BaseTestBuildRun, DockerBuild_FlavorTestBuildRun
from exaslct_src.lib.export_container_task import ExportContainerTask
from exaslct_src.lib.flavor_task import FlavorWrapperTask
from exaslct_src.release_type import ReleaseType


class ExportContainer_Release(ExportContainerTask):
    def get_release_task(self, flavor_path):
        return DockerBuild_Release(flavor_path)

    def get_release_type(self):
        return ReleaseType.Release


class ExportContainer_BaseTest(ExportContainerTask):
    def get_release_task(self, flavor_path):
        return DockerBuild_BaseTestBuildRun(flavor_path)

    def get_release_type(self):
        return ReleaseType.BaseTest


class ExportContainer_FlavorTest(ExportContainerTask):
    def get_release_task(self, flavor_path):
        return DockerBuild_FlavorTestBuildRun(flavor_path)

    def get_release_type(self):
        return ReleaseType.FlavorTest


class ExportContainer(FlavorWrapperTask):
    release_types = luigi.ListParameter(["Release"])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actual_release_types = [ReleaseType[release_type] for release_type in self.release_types]

    def requires(self):
        return [self.generate_tasks_for_flavor(flavor_path) for flavor_path in self.actual_flavor_paths]

    def generate_tasks_for_flavor(self, flavor_path):
        result = []
        if ReleaseType.Release in self.actual_release_types:
            result.append(ExportContainer_Release(flavor_path=flavor_path))
        if ReleaseType.BaseTest in self.actual_release_types:
            result.append(ExportContainer_BaseTest(flavor_path=flavor_path))
        if ReleaseType.FlavorTest in self.actual_release_types:
            result.append(ExportContainer_FlavorTest(flavor_path=flavor_path))
        return result
