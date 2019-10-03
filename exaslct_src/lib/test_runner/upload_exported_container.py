import pathlib

import luigi

from exaslct_src.lib.base.json_pickle_parameter import JsonPickleParameter
from exaslct_src.lib.data.dependency_collector.dependency_release_info_collector import DependencyExportInfoCollector
from exaslct_src.lib.data.release_info import ExportInfo
from exaslct_src.lib.test_runner.upload_file_to_db import UploadFileToBucketFS


class UploadExportedContainer(UploadFileToBucketFS):
    release_name = luigi.Parameter()
    release_goal = luigi.Parameter()
    export_info = JsonPickleParameter(ExportInfo, significant=False)

    def get_log_file(self):
        return "/exa/logs/cored/bucketfsd*"

    def get_pattern_to_wait_for(self):
        return self.export_info.name + ".*extracted"

    def get_file_to_upload(self):
        return "/exports/" + pathlib.Path(self.export_info.cache_file).name # TODO directory /exports is as data dependency to SpawnTestContainer

    def get_upload_target(self):
        return "myudfs/" + self.export_info.name + ".tar.gz"

    def get_sync_time_estimation(self) -> int:
        return 1*60