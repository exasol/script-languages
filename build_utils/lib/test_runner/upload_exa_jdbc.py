import luigi


class UploadExaJDBC(luigi.Task):
    database_info = luigi.DictParameter()
    test_container_info = luigi.DictParameter()
    pass