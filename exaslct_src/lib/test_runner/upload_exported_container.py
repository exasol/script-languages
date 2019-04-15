import pathlib

import luigi

from exaslct_src.lib.data.dependency_collector.dependency_release_info_collector import DependencyReleaseInfoCollector
from exaslct_src.lib.data.release_info import ExportInfo
from exaslct_src.lib.test_runner.upload_file_to_db import UploadFileToBucketFS


class UploadExportedContainer(UploadFileToBucketFS):
    release_name = luigi.Parameter()
    release_type = luigi.Parameter()
    release_info_dict = luigi.DictParameter(significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release_info = ExportInfo.from_dict(self.release_info_dict)

    def get_log_file(self):
        return "/exa/logs/cored/bucketfsd*"

    def get_pattern_to_wait_for(self):
        return self.release_info.name+".*extracted"

    def get_file_to_upload(self):
        return "/exports/" + pathlib.Path(self.release_info.cache_file).name # TODO directory /exports is as data dependency to SpawnTestContainer

    def get_upload_target(self):
        return "myudfs/"+self.release_info.name+".tar.gz"
