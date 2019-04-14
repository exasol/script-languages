import luigi

from exaslct_src.export_container import ExportContainer_Release, ExportContainer_BaseTest, ExportContainer_FlavorTest
from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.flavor_task import FlavorTask
from exaslct_src.lib.upload_container_task import UploadContainerTask
from exaslct_src.release_type import ReleaseType


class UploadContainer_Release(UploadContainerTask):
    def get_export_task(self, flavor_path):
        return ExportContainer_Release(flavor_path)

    def get_release_type(self):
        return ReleaseType.Release


class UploadContainer_BaseTest(UploadContainerTask):
    def get_export_task(self, flavor_path):
        return ExportContainer_BaseTest(flavor_path)

    def get_release_type(self):
        return ReleaseType.BaseTest


class UploadContainer_FlavorTest(UploadContainerTask):
    def get_export_task(self, flavor_path):
        return ExportContainer_FlavorTest(flavor_path)

    def get_release_type(self):
        return ReleaseType.FlavorTest


class UploadContainer(FlavorTask):
    release_types = luigi.ListParameter(["Release"])
    database_host = luigi.Parameter()
    bucketfs_port = luigi.IntParameter()
    bucketfs_username = luigi.Parameter()
    bucketfs_password = luigi.Parameter()
    bucketfs_name = luigi.Parameter()
    bucket_name = luigi.Parameter()
    path_in_bucket = luigi.Parameter()
    bucketfs_https = luigi.BoolParameter(False)
    release_name = luigi.OptionalParameter()

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
                         database_host=self.database_host,
                         bucketfs_port=self.bucketfs_port,
                         bucketfs_username=self.bucketfs_username,
                         bucketfs_password=self.bucketfs_password,
                         bucket_name=self.bucket_name,
                         path_in_bucket=self.path_in_bucket,
                         bucketfs_https=self.bucketfs_https,
                         release_name=self.release_name,
                         bucketfs_name=self.bucketfs_name)
        if ReleaseType.Release in self.actual_release_types:
            result.append(UploadContainer_Release(**parameter))
        if ReleaseType.BaseTest in self.actual_release_types:
            result.append(UploadContainer_BaseTest(**parameter))
        if ReleaseType.FlavorTest in self.actual_release_types:
            result.append(UploadContainer_FlavorTest(**parameter))
        return result

    def _prepare_outputs(self):
        self._target = luigi.LocalTarget(
            "%s/logs/uploads/current"
            % (self._build_config.output_directory))
        if self._target.exists():
            self._target.remove()

    def output(self):
        return self._target

    def run(self):
        with self.output().open("w") as out_file:
            for releases in self.input():
                for in_target in releases:
                    with in_target.open("r") as in_file:
                        out_file.write(in_file.read())
                        out_file.write("\n")
                        out_file.write("=================================================")
                        out_file.write("\n")

