import importlib

import luigi

from exaslct_src.test_environment.lib.base.json_pickle_parameter import JsonPickleParameter
from exaslct_src.test_environment.lib.docker.images.push.docker_image_push_base_task import DockerPushImageBaseTask
from exaslct_src.test_environment.lib.docker.images.required_task_info import RequiredTaskInfo


class DockerPushImageTask(DockerPushImageBaseTask):
    # We need to create the DockerCreateImageTask for DockerPushImageTask dynamically,
    # because we want to push as soon as possible after an image was build and
    # don't want to wait for the push finishing before starting to build depended images,
    # but we also need to create a DockerPushImageTask for each DockerCreateImageTask of a goal

    required_task_info = JsonPickleParameter(RequiredTaskInfo,
                                             visibility=luigi.parameter.ParameterVisibility.HIDDEN,
                                             significant=True)  # type:RequiredTaskInfo

    def get_docker_image_task(self):
        module = importlib.import_module(self.required_task_info.module_name)
        class_ = getattr(module, self.required_task_info.class_name)
        instance = self.create_child_task(class_, **self.required_task_info.params)
        return instance
