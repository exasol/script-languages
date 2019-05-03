from typing import Dict

from exaslct_src.lib.data.required_task_info import RequiredTaskInfo
from exaslct_src.lib.docker.docker_create_image_task import DockerCreateImageTask
from exaslct_src.lib.upload_container_task import UploadContainerTask


class UploadContainerTasksCreator():

    def __init__(self,
                 database_host: str,
                 bucketfs_port: int,
                 bucketfs_username: str,
                 bucketfs_password: str,
                 bucketfs_name: str,
                 bucket_name: str,
                 path_in_bucket: str,
                 bucketfs_https: bool,
                 release_name: str):
        self.bucketfs_https = bucketfs_https
        self.path_in_bucket = path_in_bucket
        self.bucket_name = bucket_name
        self.bucketfs_name = bucketfs_name
        self.bucketfs_password = bucketfs_password
        self.bucketfs_username = bucketfs_username
        self.bucketfs_port = bucketfs_port
        self.database_host = database_host
        self.release_name = release_name

    def create_upload_tasks_for_flavors(
            self, build_tasks: Dict[str, Dict[str, DockerCreateImageTask]]):
        return {flavor_path: self.create_upload_tasks(flavor_path, build_task)
                for flavor_path, build_task in build_tasks.items()}

    def create_upload_tasks(self, flavor_path: str,
                            build_tasks: Dict[str, DockerCreateImageTask]):
        return {release_type: self.create_upload_task(release_type, flavor_path, build_task)
                for release_type, build_task in build_tasks.items()}

    def create_upload_task(self, release_type: str, flavor_path: str,
                           build_task: DockerCreateImageTask):
        required_task_info = self.create_required_task_info(build_task)
        return \
            UploadContainerTask(
                required_task_info_json=required_task_info.to_json(indent=None),
                release_name=self.release_name,
                release_type=release_type,
                flavor_path=flavor_path,
                database_host=self.database_host,
                bucketfs_port=self.bucketfs_port,
                bucketfs_username=self.bucketfs_username,
                bucketfs_password=self.bucketfs_password,
                bucketfs_name=self.bucketfs_name,
                bucket_name=self.bucket_name,
                path_in_bucket=self.path_in_bucket,
                bucketfs_https=self.bucketfs_https
            )

    def create_required_task_info(self, build_task):
        required_task_info = \
            RequiredTaskInfo(module_name=build_task.__module__,
                             class_name=build_task.__class__.__name__,
                             params=build_task.param_kwargs)
        return required_task_info
