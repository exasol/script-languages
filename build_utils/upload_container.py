import luigi

from build_utils.export_container import ExportContainer_Release, ExportContainer_BaseTest, ExportContainer_FlavorTest
from build_utils.lib.flavor_task import FlavorWrapperTask
from build_utils.lib.upload_container_task import UploadContainerTask
from build_utils.release_type import ReleaseType


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


class UploadContainer(FlavorWrapperTask):
    release_type = luigi.Parameter(["Release"])
    database_host: luigi.Parameter()
    bucketfs_port: luigi.IntParameter()
    bucketfs_username: luigi.Parameter()
    bucketfs_password: luigi.Parameter()
    bucketfs_name: luigi.Parameter()
    bucket_name: luigi.Parameter()
    path_in_bucket: luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actual_release_types = [ReleaseType[release_type] for release_type in self.release_types]

    def requires(self):
        return [self.generate_tasks_for_flavor(flavor_path) for flavor_path in self.actual_flavor_paths]

    def generate_tasks_for_flavor(self, flavor_path):
        result = []
        if ReleaseType.Release in self.actual_release_types:
            result.append(
                UploadContainer_Release(
                    flavor_path=flavor_path,
                    database_host=self.database_host,
                    bucketfs_port=self.bucketfs_port,
                    bucketfs_username=self.bucketfs_username,
                    bucketfs_password=self.bucketfs_password,
                    bucketfs_name=self.bucketfs_name,
                    bucket_name=self.bucket_name,
                    path_in_bucket=self.path_in_bucket
                ))
        if ReleaseType.BaseTest in self.actual_release_types:
            result.append(
                UploadContainer_BaseTest(flavor_path=flavor_path))
        if ReleaseType.FlavorTest in self.actual_release_types:
            result.append(
                UploadContainer_FlavorTest(flavor_path=flavor_path))
        return result
