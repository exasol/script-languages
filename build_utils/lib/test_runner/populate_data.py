import luigi


class PopulateData(luigi.WrapperTask):
    test_environment_info_dict = luigi.DictParameter()
