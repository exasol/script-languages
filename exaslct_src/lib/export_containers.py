import jsonpickle
import luigi

from exaslct_src.lib.docker_flavor_build_base import DockerFlavorBuildBase
from exaslct_src.lib.export_container_tasks_creator import ExportContainerTasksCreator
from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_release_info_collector import DependencyExportInfoCollector
from exaslct_src.lib.release_type import ReleaseType


class ExportContainers(DockerFlavorBuildBase):
    command_line_output_target = \
        luigi.LocalTarget("%s/exports/command_line_output" % (build_config().output_directory))

    release_types = luigi.ListParameter(["Release"])
    export_path = luigi.OptionalParameter(None)
    release_name = luigi.OptionalParameter(None)
    # TOOD force export

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_outputs()

    def get_goals(self):
        self.actual_release_types = [ReleaseType[release_type] for release_type in self.release_types]
        release_type_goal_map = {ReleaseType.Release: "release",
                                 ReleaseType.BaseTest: "base_test_build_run",
                                 ReleaseType.FlavorTest: "flavor_test_build_run"}
        goals = [release_type_goal_map[release_type] for release_type in self.actual_release_types]
        return goals

    def _prepare_outputs(self):
        self._export_info_target = luigi.LocalTarget(
            "%s/info/exports/final" % (build_config().output_directory))
        if self._export_info_target.exists():
            self._export_info_target.remove()

    def output(self):
        return self._export_info_target

    def run_task(self):
        build_tasks = self.create_build_tasks_for_all_flavors(not build_config().force_rebuild)
        tasks_creator = ExportContainerTasksCreator(export_path=self.export_path,
                                                    release_name=self.release_name)
        export_tasks = tasks_creator.create_export_tasks_for_flavors(build_tasks)
        exports_for_flavors = yield export_tasks
        self.write_command_line_output(exports_for_flavors)
        self.write_output(exports_for_flavors)

    def write_output(self, exports_for_flavors):
        result = {}
        for flavor_path, releases in exports_for_flavors.items():
            result[flavor_path] = {}
            for release_name, release_info_dict in releases.items():
                export_info = DependencyExportInfoCollector().get_from_sinlge_input(release_info_dict)
                result[flavor_path][release_name]=export_info
        with self.output().open("w") as f:
            json = self.json_pickle_result(result)
            f.write(json)

    def json_pickle_result(self, result):
        jsonpickle.set_preferred_backend('simplejson')
        jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)
        json = jsonpickle.encode(result)
        return json

    def write_command_line_output(self, exports_for_flavors):
        if self.command_line_output_target.exists():
            self.command_line_output_target.remove()
        with self.command_line_output_target.open("w") as out_file:
            for flavor_path, releases in exports_for_flavors.items():
                for release_name, release_info_dict in releases.items():
                    export_info = DependencyExportInfoCollector().get_from_sinlge_input(release_info_dict)
                    out_file.write("Cached container under %s" % export_info.cache_file)
                    out_file.write("\n")
                    out_file.write("\n")
                    if export_info.output_file is not None:
                        out_file.write("Copied container to %s" % export_info.output_file)
                        out_file.write("\n")
                        out_file.write("\n")
                    out_file.write("=================================================")
                    out_file.write("\n")

