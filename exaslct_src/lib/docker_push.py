import importlib

from exaslct_src.lib.docker_build import *
from exaslct_src.lib.data.required_task_info import RequiredTaskInfo
from exaslct_src.lib.docker.docker_create_image_task import DockerCreateImageTask
from exaslct_src.lib.docker.docker_push_task import DockerPushImageBaseTask
from exaslct_src.lib.docker_flavor_build_base import DockerFlavorBuildBase
from exaslct_src.lib.task_creator_from_build_tasks import TaskCreatorFromBuildTasks


class DockerPushImageTask(DockerPushImageBaseTask):
    # We need to create the DockerCreateImageTask for DockerPushImageTask dynamically,
    # because we want to push as soon as possible after an image was build and
    # don't want to wait for the push finishing before starting to build depended images,
    # but we also need to create a DockerPushImageTask for each DockerCreateImageTask of a goal

    required_task_info_json = luigi.Parameter(visibility=luigi.parameter.ParameterVisibility.HIDDEN,
                                              significant=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_docker_image_task(self):
        instance = self.create_required_task(self.required_task_info_json)
        return instance

    def create_required_task(self, required_task_info_json: str) -> DockerCreateImageTask:
        required_task_info = RequiredTaskInfo.from_json(required_task_info_json)
        module = importlib.import_module(required_task_info.module_name)
        class_ = getattr(module, required_task_info.class_name)
        instance = class_(**required_task_info.params)
        return instance


class DockerPush(DockerFlavorBuildBase):
    goals = luigi.ListParameter()
    force_push = luigi.BoolParameter(False)
    push_all = luigi.BoolParameter(False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_outputs()

    def get_goals(self):
        return self.goals

    def _prepare_outputs(self):
        self._push_info_target = luigi.LocalTarget(
            "%s/info/push/final" % (build_config().output_directory))
        if self._push_info_target.exists():
            self._push_info_target.remove()

    def output(self):
        return self._push_info_target

    def run_task(self):
        build_tasks_for_flavors = self.create_build_tasks_for_all_flavors(shortcut_build=not self.push_all)
        push_task_creator = PushTaskCreatorFromBuildTasks(self.force_push)
        push_tasks = push_task_creator.create_tasks_for_flavors(build_tasks_for_flavors)
        pushs_for_flavors = yield push_tasks
        self.write_output(pushs_for_flavors)

    def write_output(self, result):
        with self.output().open("w") as f:
            json = self.json_pickle_result(result)
            f.write(json)

    def json_pickle_result(self, result):
        jsonpickle.set_preferred_backend('simplejson')
        jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)
        json = jsonpickle.encode(result)
        return json

class PushTaskCreatorFromBuildTasks(TaskCreatorFromBuildTasks):

    def __init__(self, force_push:bool):
        self.force_push = force_push

    def create_task_with_required_tasks(self, build_task, required_task_info):
        push_task = \
            DockerPushImageTask(
                image_name=build_task.image_name,
                required_task_info_json=required_task_info.to_json(indent=None),
                force_push=self.force_push)
        return push_task
