from typing import Dict

from exaslct_src.lib.data.dependency_collector.dependency_collector import DependencyInfoCollector
from exaslct_src.lib.data.required_task_info import RequiredTaskInfo


class DependencyRequiredTaskInfoCollector(DependencyInfoCollector[RequiredTaskInfo]):

    def is_info(self, input):
        return isinstance(input, Dict) and REQUIRED_TASK_INFO in input

    def read_info(self, value) -> RequiredTaskInfo:
        with value[REQUIRED_TASK_INFO].open("r") as file:
            return RequiredTaskInfo.from_json(file.read())


REQUIRED_TASK_INFO = "required_task_info"
