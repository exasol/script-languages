from typing import Dict

from exasol_integration_test_docker_environment.lib.docker.images.create.docker_image_create_task import \
    DockerCreateImageTask
from exasol_integration_test_docker_environment.lib.docker.images.required_task_info import RequiredTaskInfo

from exaslct_src.exaslct.lib.tasks.upload.upload_container_task import UploadContainerTask
from exaslct_src.exaslct.lib.tasks.upload.upload_containers_parameter import UploadContainersParameter


class UploadContainerTasksCreator:

    def __init__(self, task: UploadContainersParameter):
        self.task = task

    def create_upload_tasks(self, build_tasks: Dict[str, DockerCreateImageTask]):
        return {release_goal: self._create_upload_task(release_goal, build_task)
                for release_goal, build_task in build_tasks.items()}

    def _create_upload_task(self, release_goal: str, build_task: DockerCreateImageTask):
        required_task_info = self._create_required_task_info(build_task)
        return self.task.create_child_task_with_common_params(
            UploadContainerTask,
            required_task_info=required_task_info,
            release_goal=release_goal
        )

    def _create_required_task_info(self, build_task):
        required_task_info = \
            RequiredTaskInfo(module_name=build_task.__module__,
                             class_name=build_task.__class__.__name__,
                             params=build_task.param_kwargs)
        return required_task_info
