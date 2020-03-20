from exaslct_src.test_environment.base.base_task import BaseTask
from exaslct_src.test_environment.docker.images.push.docker_image_push_task import DockerPushImageTask
from exaslct_src.test_environment.docker.images.task_creator_from_build_tasks import TaskCreatorFromBuildTasks


class PushTaskCreatorFromBuildTasks(TaskCreatorFromBuildTasks):

    def __init__(self, task: BaseTask):
        self.task = task

    def create_task_with_required_tasks(self, build_task, required_task_info):
        push_task = \
            self.task.create_child_task_with_common_params(
                DockerPushImageTask,
                image_name=build_task.image_name,
                required_task_info=required_task_info)
        return push_task
