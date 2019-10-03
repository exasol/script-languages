from exaslct_src.lib.base.base_task import BaseTask
from exaslct_src.lib.docker_save_image_task import DockerSaveImageTask
from exaslct_src.lib.task_creator_from_build_tasks import TaskCreatorFromBuildTasks


class SaveTaskCreatorFromBuildTasks(TaskCreatorFromBuildTasks):

    def __init__(self, task: BaseTask):
        self.task = task

    def create_task_with_required_tasks(self, build_task, required_task_info):
        push_task = \
            self.task.create_child_task_with_common_params(
                DockerSaveImageTask,
                image_name=build_task.image_name,
                required_task_info=required_task_info,

            )
        return push_task
