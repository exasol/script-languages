from typing import Dict

from exaslct_src.exaslct.lib.tasks.build.docker_flavor_build_base import DockerFlavorBuildBase
from exaslct_src.exaslct.lib.tasks.save.docker_save_parameter import DockerSaveParameter
from exaslct_src.test_environment.src.lib.base.flavor_task import FlavorsBaseTask
from exaslct_src.test_environment.src.lib.docker.images.save.save_task_creator_for_build_tasks import \
    SaveTaskCreatorFromBuildTasks


class DockerSave(FlavorsBaseTask, DockerSaveParameter):
    def register_required(self):
        tasks = self.create_tasks_for_flavors_with_common_params(
            DockerFlavorSave)  # type: Dict[str,DockerFlavorSave]
        self.image_info_futures = self.register_dependencies(tasks)

    def run_task(self):
        image_infos = self.get_values_from_futures(self.image_info_futures)
        self.return_object(image_infos)


class DockerFlavorSave(DockerFlavorBuildBase, DockerSaveParameter):

    def get_goals(self):
        return self.goals

    def run_task(self):
        build_tasks = self.create_build_tasks(shortcut_build=not self.save_all)
        save_task_creator = SaveTaskCreatorFromBuildTasks(self)
        save_tasks = save_task_creator.create_tasks_for_build_tasks(build_tasks)
        image_info_furtures = yield from self.run_dependencies(save_tasks)
        image_infos = self.get_values_from_futures(image_info_furtures)
        self.return_object(image_infos)
