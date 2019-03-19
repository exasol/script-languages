import luigi


class PopulateData(luigi.Task):
    database_info = luigi.DictParameter()
    db = luigi.DictParameter()
    pass