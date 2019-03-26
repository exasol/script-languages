from build_utils.lib.test_runner.upload_file_to_db import UploadFileToDB


class UploadExaJDBC(UploadFileToDB):

    def get_log_file(self):
        return "/exa/logs/cored/*bucketfsd*"

    def get_pattern_to_wait_for(self):
        return "exajdbc.jar"

    def get_file_to_upload(self):
        return "downloads/JDBC/exajdbc.jar"

    def get_upload_target(self):
        return "jdbc-adapter/exajdbc.jar"
