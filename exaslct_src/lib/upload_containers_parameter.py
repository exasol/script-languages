import luigi

from exaslct_src.lib.upload_container_parameter import UploadContainerParameter


class UploadContainersParameter(UploadContainerParameter):
    release_goals = luigi.ListParameter(["release"])