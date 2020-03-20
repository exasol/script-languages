import luigi

from exaslct_src.exaslct.lib.tasks.upload.upload_container_parameter import UploadContainerParameter


class UploadContainersParameter(UploadContainerParameter):
    release_goals = luigi.ListParameter(["release"])
