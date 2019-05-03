import logging
from pathlib import Path

import luigi
import requests
from jinja2 import Template

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_release_info_collector import DependencyExportInfoCollector, \
    EXPORT_INFO
from exaslct_src.lib.data.release_info import ExportInfo
from exaslct_src.lib.docker_config import docker_config
from exaslct_src.lib.flavor import flavor
from exaslct_src.lib.stoppable_task import StoppableTask

# TODO check if upload was successfull by requesting the file
# TODO add error checks and propose reasons for the error
# TODO extract bucketfs interaction into own module
class UploadContainerBaseTask(StoppableTask):
    logger = logging.getLogger('luigi-interface')
    flavor_path = luigi.Parameter()
    release_name = luigi.OptionalParameter()
    release_type = luigi.Parameter()
    database_host = luigi.Parameter()
    bucketfs_port = luigi.IntParameter()
    bucketfs_username = luigi.Parameter()
    bucketfs_password = luigi.Parameter(significant=False, visibility=luigi.parameter.ParameterVisibility.HIDDEN)
    bucketfs_https = luigi.BoolParameter(False)
    bucketfs_name = luigi.Parameter()
    bucket_name = luigi.Parameter()
    path_in_bucket = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._target = luigi.LocalTarget(
            "%s/uploads/%s/%s"
            % (build_config().output_directory,
               flavor.get_name_from_path(self.flavor_path),
               self.release_type))
        if self._target.exists():
            self._target.remove()

    def output(self):
        return self._target

    def requires_tasks(self):
        return {"export": self.get_export_task()}

    def get_export_task(self):
        pass

    def run_task(self):
        release_info = self.get_release_info()
        self.upload_container(release_info)
        language_definition = self.generate_language_definition(release_info)
        self.write_output(language_definition, release_info)

    def write_output(self, language_definition, release_info):
        flavor_name = flavor.get_name_from_path(self.flavor_path)
        try:
            release_path = Path(release_info.cache_file).relative_to(Path(".").absolute())
        except ValueError as e:
            release_path = Path(release_info.cache_file)
        with self.output().open("w") as f:
            f.write(f"""Uploaded {release_path} to \n""" + \
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
        language_definition_path = Path(self.flavor_path).joinpath("flavor_base").joinpath("language_definition")
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
        self.logger.info(f"Upload to {release_info.cache_file} to {self.get_upload_url(release_info, without_login=True)}")
        with open(release_info.cache_file, 'rb') as file:
            r = s.put(url, data=file)

    def get_upload_url(self, release_info: ExportInfo, without_login: bool = False):
        complete_release_name = self.get_complete_release_name(release_info)
        if without_login:
            login = ""
        else:
            login = f"""{self.bucketfs_username}:{self.bucketfs_password}@"""
        url = f"""{self.get_url_prefix()}{login}""" + \
              f"""{self.database_host}:{self.bucketfs_port}/{self.bucket_name}/{self.path_in_bucket}/""" + \
              complete_release_name + ".tar.gz"
        return url

    def get_complete_release_name(self, release_info: ExportInfo):
        complete_release_name = f"""{release_info.name}-{release_info.release_type}-{self.get_release_name(
            release_info)}"""
        return complete_release_name

    def get_release_info(self):
        release_info_of_dependencies = \
            DependencyExportInfoCollector().get_from_dict_of_inputs(self.input())
        release_info = release_info_of_dependencies["export"]
        return release_info
