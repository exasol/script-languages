import logging
from datetime import datetime
from pathlib import Path

import luigi
import requests
from jinja2 import Template

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_release_info_collector import RELEASE_INFO, \
    DependencyReleaseInfoCollector
from exaslct_src.lib.data.release_info import ExportInfo
from exaslct_src.lib.docker_config import docker_config
from exaslct_src.lib.flavor import flavor
from exaslct_src.release_type import ReleaseType
from exaslct_src.stoppable_task import StoppableTask


class UploadContainerTask(StoppableTask):
    logger = logging.getLogger('luigi-interface')
    flavor_path = luigi.Parameter()
    release_name = luigi.OptionalParameter()
    database_host = luigi.Parameter()
    bucketfs_port = luigi.IntParameter()
    bucketfs_username = luigi.Parameter()
    bucketfs_password = luigi.Parameter(significant=False, visibility=luigi.parameter.ParameterVisibility.PRIVATE)
    bucketfs_https = luigi.BoolParameter(False)
    bucketfs_name = luigi.Parameter()
    bucket_name = luigi.Parameter()
    path_in_bucket = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._docker_config = docker_config()
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._target = luigi.LocalTarget(
            "%s/uploads/%s/%s"
            % (self._build_config.output_directory,
               flavor.get_name_from_path(self.flavor_path),
               self.get_release_type().name))
        if self._target.exists():
            self._target.remove()

    def output(self):
        return self._target

    def requires(self):
        return {"release": self.get_export_task(self.flavor_path)}

    def get_export_task(self, flavor_path):
        pass

    def get_release_type(self) -> ReleaseType:
        pass

    def run_task(self):
        release_info = self.get_release_info()
        self.upload_container(release_info)
        language_definition = self.generate_language_definition(release_info)
        self.write_output(language_definition, release_info)

    def write_output(self, language_definition, release_info):
        flavor_name = flavor.get_name_from_path(self.flavor_path)
        relative_release_path = Path(release_info.path).relative_to(Path(".").absolute())
        with self.output().open("w") as f:
            f.write(f"""Uploaded {relative_release_path} to \n""" + \
                    f"""{self.get_upload_url(release_info, without_login=True)}""")
            f.write("\n")
            f.write("\n")
            f.write(f"""In SQL you can activate the languages supported by the {flavor_name} flavor """ + \
                    f"""by using a statement like this:""")
            f.write("\n")
            f.write("\n")
            f.write(f"""ALTER SESSION SET SCRIPT_LANGUAGES='{language_definition}';""")
            f.write("\n")

    def generate_language_definition(self, release_info: ExportInfo):
        language_definition_path = Path(self.flavor_path).joinpath("language_definition")
        with language_definition_path.open("r") as f:
            language_definition_template = f.read()
        template = Template(language_definition_template)
        language_definition = template.render(bucketfs_name=self.bucketfs_name,
                                              bucket_name=self.bucket_name,
                                              path_in_bucket=self.path_in_bucket,
                                              release_name=self.get_complete_release_name(release_info))
        return language_definition

    def get_url_prefix(self):
        if self.bucketfs_https:
            url_prefix = "https://"
        else:
            url_prefix = "http://"
        return url_prefix

    def get_release_name(self, release_info: ExportInfo):
        if self.release_name is None:
            release_name = release_info.hash
        else:
            release_name = self.release_name
        return release_name

    def upload_container(self, release_info: ExportInfo):
        s = requests.session()
        url = self.get_upload_url(release_info)
        with open(release_info.path, 'rb') as file:
            r = s.put(url, data=file)

    def get_upload_url(self, release_info: ExportInfo, without_login: bool = False):
        complete_release_name = self.get_complete_release_name(release_info)
        if without_login:
            login = ""
        else:
            login = f"""{self.bucketfs_username}:{self.bucketfs_password}@"""
        url = f"""{self.get_url_prefix()}{login}""" + \
              f"""{self.database_host}:{self.bucketfs_port}/{self.bucket_name}/{self.path_in_bucket}/""" + \
              complete_release_name
        return url

    def get_complete_release_name(self, release_info: ExportInfo):
        complete_release_name = f"""{release_info.name}-{release_info.release_type.name}-{self.get_release_name(
            release_info)}"""
        return complete_release_name

    def get_release_info(self):
        release_info_of_dependencies = \
            DependencyReleaseInfoCollector().get_from_dict_of_inputs(self.input())
        release_info = release_info_of_dependencies["release"]
        return release_info
