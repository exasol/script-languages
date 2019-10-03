import importlib

import luigi

from exaslct_src.lib.base.json_pickle_parameter import JsonPickleParameter
from exaslct_src.lib.data.required_task_info import RequiredTaskInfo
from exaslct_src.lib.docker.docker_create_image_task import DockerCreateImageTask
from exaslct_src.lib.export_container_base_task import ExportContainerBaseTask
from exaslct_src.lib.upload_container_base_task import UploadContainerBaseTask


class UploadContainerTask(UploadContainerBaseTask):
    # We need to create the ExportContainerTask for UploadContainerTask dynamically,
    # because we want to push as soon as possible after an image was build and
    # don't want to wait for the push finishing before starting the build of depended images,
    # but we also need to create a UploadContainerTask for each ExportContainerTask of a goal

    required_task_info = JsonPickleParameter(RequiredTaskInfo,
                                             visibility=luigi.parameter.ParameterVisibility.HIDDEN,
                                             significant=True)  # type: RequiredTaskInfo

    def get_export_task(self):
        module = importlib.import_module(self.required_task_info.module_name)
        class_ = getattr(module, self.required_task_info.class_name)
        instance = self.create_child_task(class_, **self.required_task_info.params)
        return instance
