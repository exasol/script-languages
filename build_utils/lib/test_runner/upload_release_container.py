import pathlib

import luigi

from build_utils.lib.data.release_info import ReleaseInfo
from build_utils.lib.test_runner.upload_file_to_db import UploadFileToDB


class UploadReleaseContainer(UploadFileToDB):
    release_info_dict = luigi.DictParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release_info = ReleaseInfo.from_dict(self.release_info_dict)

    def get_log_file(self):
        return "/exa/logs/cored/*bucketfsd*"

    def get_pattern_to_wait_for(self):
        return self.release_info.name

    def get_file_to_upload(self):
        return "/releases/" + pathlib.Path(self.release_info.path).name

    def get_upload_target(self):
        return self.release_info.name
