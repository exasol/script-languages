import luigi

from exaslct_src.docker_build import DockerBuild_Release, DockerBuild_BaseTestBuildRun, DockerBuild_FlavorTestBuildRun
from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_release_info_collector import DependencyReleaseInfoCollector
from exaslct_src.lib.export_container_task import ExportContainerTask
from exaslct_src.lib.flavor_task import FlavorWrapperTask, FlavorTask
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


class ExportContainer(FlavorTask):
    release_types = luigi.ListParameter(["Release"])
    output_path = luigi.OptionalParameter(None)
    release_name = luigi.OptionalParameter(None)


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._prepare_outputs()
        self.actual_release_types = [ReleaseType[release_type] for release_type in self.release_types]

    def requires(self):
        return [self.generate_tasks_for_flavor(flavor_path) for flavor_path in self.actual_flavor_paths]

    def generate_tasks_for_flavor(self, flavor_path):
        result = []
        parameter = dict(flavor_path=flavor_path,
                         output_path=self.output_path,
                         release_name=self.release_name)
        if ReleaseType.Release in self.actual_release_types:
            result.append(ExportContainer_Release(**parameter))
        if ReleaseType.BaseTest in self.actual_release_types:
            result.append(ExportContainer_BaseTest(**parameter))
        if ReleaseType.FlavorTest in self.actual_release_types:
            result.append(ExportContainer_FlavorTest(**parameter))
        return result

    def _prepare_outputs(self):
        self._target = luigi.LocalTarget(
            "%s/exports/current"
            % (self._build_config.output_directory))
        if self._target.exists():
            self._target.remove()

    def output(self):
        return self._target

    def run_task(self):
        with self.output().open("w") as out_file:
            for releases in self.input():
                for release in releases:
                    export_info = DependencyReleaseInfoCollector().get_from_sinlge_input(release)
                    out_file.write("Cached container under %s"%export_info.cache_file)
                    if export_info.output_file is not None:
                        out_file.write("\n")
                        if export_info.output_file is not None:
                            out_file.write("Copied container to %s" % export_info.output_file)
                            out_file.write("\n")
                    out_file.write("=================================================")
                    out_file.write("\n")

