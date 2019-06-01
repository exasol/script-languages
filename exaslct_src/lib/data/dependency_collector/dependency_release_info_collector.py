from typing import Dict

from exaslct_src.lib.data.release_info import ExportInfo
from exaslct_src.lib.data.dependency_collector.dependency_collector import DependencyInfoCollector


class DependencyExportInfoCollector(DependencyInfoCollector[ExportInfo]):

    def is_info(self, input):
        return isinstance(input, Dict) and EXPORT_INFO in input

    def read_info(self, value) -> ExportInfo:
        with value[EXPORT_INFO].open("r") as file:
            return ExportInfo.from_json(file.read())


EXPORT_INFO = "export_info"
