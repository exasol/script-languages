from exaslct_src.test_environment.lib.upload_file_to_db import UploadFileToBucketFS


class UploadVirtualSchemaJDBCAdapter(UploadFileToBucketFS):

    def get_log_file(self):
        return "/exa/logs/cored/*bucketfsd*"

    def get_pattern_to_wait_for(self):
        return "virtualschema-jdbc-adapter.jar.*linked"

    def get_file_to_upload(self):
        return "downloads/virtualschema-jdbc-adapter/virtualschema-jdbc-adapter.jar"

    def get_upload_target(self):
        return "jdbc_adapter/virtualschema-jdbc-adapter.jar"

    def get_sync_time_estimation(self) -> int:
        return 10
