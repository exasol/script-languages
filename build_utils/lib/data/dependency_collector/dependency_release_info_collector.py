from typing import Dict

from build_utils.lib.data.release_info import ReleaseInfo
from build_utils.lib.data.dependency_collector.dependency_collector import DependencyInfoCollector


class DependencyReleaseInfoCollector(DependencyInfoCollector[ReleaseInfo]):

    def is_info(self, input):
        return isinstance(input, Dict) and RELEASE_INFO in input

    def read_info(self, value) -> ReleaseInfo:
        with value[RELEASE_INFO].open("r") as file:
            return ReleaseInfo.from_json(file.read())


RELEASE_INFO = "release_info"
