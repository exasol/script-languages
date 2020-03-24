import pathlib

import luigi

from exaslct_src.exaslct.lib.tasks.export.export_info import ExportInfo
from exaslct_src.test_environment.src.lib.base.json_pickle_parameter import JsonPickleParameter
from exaslct_src.test_environment.src.lib.test_environment.upload_file_to_db import UploadFileToBucketFS


class UploadExportedContainer(UploadFileToBucketFS):
    release_name = luigi.Parameter()
    release_goal = luigi.Parameter()
    export_info = JsonPickleParameter(ExportInfo, significant=False)  # type: ExportInfo

    def get_log_file(self):
        return "/exa/logs/cored/bucketfsd*"

    def get_pattern_to_wait_for(self):
        return self.export_info.name + ".*extracted"

    def get_file_to_upload(self):
        file = "/exports/" + pathlib.Path(self.export_info.cache_file).name
        return file

    def get_upload_target(self):
        return "myudfs/" + self.export_info.name + ".tar.gz"

    def get_sync_time_estimation(self) -> int:
        return 1 * 60
