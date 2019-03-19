import luigi


class UploadReleaseContainer(luigi.Task):
    test_container_info = luigi.DictParameter()
    pass