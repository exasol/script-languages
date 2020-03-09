import luigi
from luigi import Config


class DockerPushParameter(Config):
    force_push = luigi.BoolParameter(False)
    push_all = luigi.BoolParameter(False)