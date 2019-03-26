import luigi


class PopulateData(luigi.WrapperTask):
    database_info_dict = luigi.DictParameter()
    test_container_info_dict = luigi.DictParameter()
