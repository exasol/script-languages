from exaslct_src.lib.test_environment.upload_file_to_db import UploadFileToBucketFS


class UploadExaJDBC(UploadFileToBucketFS):

    def get_log_file(self):
        return "/exa/logs/cored/*bucketfsd*"

    def get_pattern_to_wait_for(self):
        return "exajdbc.jar.*linked"

    def get_file_to_upload(self):
        return "downloads/JDBC/exajdbc.jar"

    def get_upload_target(self):
        return "jdbc_adapter/exajdbc.jar"

    def get_sync_time_estimation(self) -> int:
        return 10