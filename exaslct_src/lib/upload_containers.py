from typing import Dict

import luigi

from exaslct_src.lib.docker_flavor_build_base import DockerFlavorBuildBase
from exaslct_src.lib.export_container_tasks_creator import ExportContainerTasksCreator
from exaslct_src.lib.flavor_task import FlavorsBaseTask
from exaslct_src.lib.upload_container_tasks_creator import UploadContainerTasksCreator
from exaslct_src.lib.upload_containers_parameter import UploadContainersParameter


class UploadContainers(FlavorsBaseTask, UploadContainersParameter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        command_line_output_path = self.get_output_path().joinpath("command_line_output")
        self.command_line_output_target = luigi.LocalTarget(str(command_line_output_path))

    def register_required(self):
        tasks = self.create_tasks_for_flavors_with_common_params(
            UploadFlavorContainers)  # type: Dict[str,UploadFlavorContainers]
        self.export_info_futures = self.register_dependencies(tasks)

    def run_task(self):
        uploads = self.get_values_from_futures(
            self.export_info_futures)
        self.write_command_line_output(uploads)

    def write_command_line_output(self, uploads):
        with self.command_line_output_target.open("w") as out_file:
            for releases in uploads.values():
                for command_line_output_str in releases.values():
                    out_file.write(command_line_output_str)
                    out_file.write("\n")
                    out_file.write("=================================================")
                    out_file.write("\n")


class UploadFlavorContainers(DockerFlavorBuildBase, UploadContainersParameter):

    def get_goals(self):
        return set(self.release_goals)

    def run_task(self):
        build_tasks = self.create_build_tasks()

        export_tasks = self.create_export_tasks(build_tasks)
        upload_tasks = self.create_upload_tasks(export_tasks)

        command_line_output_string_futures = yield from self.run_dependencies(upload_tasks)
        command_line_output_strings = self.get_values_from_futures(command_line_output_string_futures)
        self.return_object(command_line_output_strings)

    def create_upload_tasks(self, export_tasks):
        upload_tasks_creator = UploadContainerTasksCreator(self)
        upload_tasks = upload_tasks_creator.create_upload_tasks(export_tasks)
        return upload_tasks

    def create_export_tasks(self, build_tasks):
        export_tasks_creator = ExportContainerTasksCreator(self, export_path=None)
        export_tasks = export_tasks_creator.create_export_tasks(build_tasks)
        return export_tasks
