import luigi
from luigi import Config


class DockerSaveParameter(Config):
    force_save = luigi.BoolParameter(False)
    save_all = luigi.BoolParameter(False)
    save_path = luigi.Parameter()
    goals = luigi.ListParameter([])