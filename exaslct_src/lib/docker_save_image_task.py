import importlib

import luigi

from exaslct_src.lib.base.json_pickle_parameter import JsonPickleParameter
from exaslct_src.lib.data.required_task_info import RequiredTaskInfo
from exaslct_src.lib.docker.docker_create_image_task import DockerCreateImageTask
from exaslct_src.lib.docker.docker_save_task import DockerSaveImageBaseTask


class DockerSaveImageTask(DockerSaveImageBaseTask):
    # We need to create the DockerCreateImageTask for DockerSaveImageTask dynamically,
    # because we want to save as soon as possible after an image was build and
    # don't want to wait for the save finishing before starting to build depended images,
    # but we also need to create a DockerSaveImageTask for each DockerCreateImageTask of a goal

    required_task_info = JsonPickleParameter(RequiredTaskInfo,
                                             visibility=luigi.parameter.ParameterVisibility.HIDDEN,
                                             significant=True)  # type: RequiredTaskInfo

    def get_docker_image_task(self):
        module = importlib.import_module(self.required_task_info.module_name)
        class_ = getattr(module, self.required_task_info.class_name)
        instance = self.create_child_task(class_, **self.required_task_info.params)
        return instance
