import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.docker_flavor_build_base import DockerFlavorBuildBase
from exaslct_src.lib.export_container_tasks_creator import ExportContainerTasksCreator
from exaslct_src.lib.release_type import ReleaseType
from exaslct_src.lib.upload_container_tasks_creator import UploadContainerTasksCreator


class UploadContainer(DockerFlavorBuildBase):
    release_types = luigi.ListParameter(["Release"])
    database_host = luigi.Parameter()
    bucketfs_port = luigi.IntParameter()
    bucketfs_username = luigi.Parameter(significant=False)
    bucketfs_password = luigi.Parameter(significant=False, visibility=luigi.parameter.ParameterVisibility.PRIVATE)
    bucketfs_name = luigi.Parameter()
    bucket_name = luigi.Parameter()
    path_in_bucket = luigi.Parameter()
    bucketfs_https = luigi.BoolParameter(False)
    release_name = luigi.OptionalParameter()

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
        self._upload_target = luigi.LocalTarget(
            "%s/uploads/command_line_output" % (build_config().output_directory))
        if self._upload_target.exists():
            self._upload_target.remove()

    def output(self):
        return self._upload_target

    def run_task(self):
        build_tasks = self.create_build_tasks_for_all_flavors()
        export_tasks_creator = ExportContainerTasksCreator(export_path=None,
                                                           release_name=self.release_name)
        export_tasks = export_tasks_creator.create_export_tasks_for_flavors(build_tasks)
        upload_tasks_creator = UploadContainerTasksCreator(database_host=self.database_host,
                                                           bucketfs_port=self.bucketfs_port,
                                                           bucketfs_username=self.bucketfs_username,
                                                           bucketfs_password=self.bucketfs_password,
                                                           bucketfs_name=self.bucketfs_name,
                                                           bucket_name=self.bucket_name,
                                                           path_in_bucket=self.path_in_bucket,
                                                           bucketfs_https=self.bucketfs_https,
                                                           release_name=self.release_name)
        upload_tasks = upload_tasks_creator.create_upload_tasks_for_flavors(export_tasks)
        uploads_of_flavors = yield upload_tasks
        self.write_command_line_output(uploads_of_flavors)

    def write_command_line_output(self, exports_for_flavors):
        print("AAAAAAAA",exports_for_flavors)
        with self._upload_target.open("w") as out_file:
            for releases in exports_for_flavors.values():
                for in_target in releases.values():
                    with in_target.open("r") as in_file:
                        out_file.write(in_file.read())
                        out_file.write("\n")
                        out_file.write("=================================================")
                        out_file.write("\n")