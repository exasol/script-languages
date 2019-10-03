from typing import Dict

from exaslct_src.lib.docker_build import *
from exaslct_src.lib.docker_flavor_build_base import DockerFlavorBuildBase
from exaslct_src.lib.save_task_creator_from_build_tasks import SaveTaskCreatorFromBuildTasks


class DockerSaveParameter(Config):
    force_save = luigi.BoolParameter(False)
    save_all = luigi.BoolParameter(False)
    save_path = luigi.Parameter()
    goals = luigi.ListParameter([])


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