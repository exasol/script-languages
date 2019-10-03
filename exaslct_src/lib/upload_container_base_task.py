import textwrap
from pathlib import Path

import luigi
import requests

from exaslct_src.AbstractMethodException import AbstractMethodException
from exaslct_src.lib.data.release_info import ExportInfo
from exaslct_src.lib.flavor import flavor
from exaslct_src.lib.flavor_task import FlavorBaseTask
from exaslct_src.lib.language_definition import LanguageDefinition
from exaslct_src.lib.upload_container_parameter import UploadContainerParameter


class UploadContainerBaseTask(FlavorBaseTask, UploadContainerParameter):
    # TODO check if upload was successfull by requesting the file
    # TODO add error checks and propose reasons for the error
    # TODO extract bucketfs interaction into own module

    release_goal = luigi.Parameter()

    def register_required(self):
        task = self.get_export_task()
        self.export_info_future = self.register_dependency(task)

    def get_export_task(self):
        raise AbstractMethodException()

    def run_task(self):
        export_info = self.get_values_from_future(self.export_info_future)
        self._upload_container(export_info)
        language_definition = \
            LanguageDefinition(release_name=self._get_complete_release_name(export_info),
                               flavor_path=self.flavor_path,
                               bucketfs_name=self.bucketfs_name,
                               bucket_name=self.bucket_name,
                               path_in_bucket=self.path_in_bucket)
        command_line_output_str = \
            self.generate_command_line_output_str(
                language_definition, export_info)
        self.return_object(command_line_output_str)

    def generate_command_line_output_str(self,
                                         language_definition: LanguageDefinition,
                                         export_info: ExportInfo):
        flavor_name = self.get_flavor_name()
        try:
            release_path = Path(export_info.cache_file).relative_to(Path(".").absolute())
        except ValueError as e:
            release_path = Path(export_info.cache_file)
        command_line_output_str = textwrap.dedent(f"""
            Uploaded {release_path} to 
            {self._get_upload_url(export_info, without_login=True)}
            
            
            In SQL, you can activate the languages supported by the {flavor_name} 
            flavor by using the following statements:
            
            
            To active the flavor only for the current session:
            
            {language_definition.generate_alter_session()}
            
            
            To active the flavor only for the current session:
            
            {language_definition.generate_alter_system()}
            """)
        return command_line_output_str

    def _upload_container(self, release_info: ExportInfo):
        s = requests.session()
        url = self._get_upload_url(release_info)
        self.logger.info(
            f"Upload to {release_info.cache_file} to {self._get_upload_url(release_info, without_login=True)}")
        with open(release_info.cache_file, 'rb') as file:
            r = s.put(url, data=file)

    def _get_upload_url(self, release_info: ExportInfo, without_login: bool = False):
        complete_release_name = self._get_complete_release_name(release_info)
        if without_login:
            login = ""
        else:
            login = f"""{self.bucketfs_username}:{self.bucketfs_password}@"""
        url = f"""{self._get_url_prefix()}{login}""" + \
              f"""{self.database_host}:{self.bucketfs_port}/{self.bucket_name}/{self.path_in_bucket}/""" + \
              complete_release_name + ".tar.gz"
        return url

    def _get_complete_release_name(self, release_info: ExportInfo):
        complete_release_name = f"""{release_info.name}-{release_info.release_goal}-{self._get_release_name(
            release_info)}"""
        return complete_release_name

    def _get_release_name(self, release_info: ExportInfo):
        if self.release_name is None:
            release_name = release_info.hash
        else:
            release_name = self.release_name
        return release_name

    def _get_url_prefix(self):
        if self.bucketfs_https:
            url_prefix = "https://"
        else:
            url_prefix = "http://"
        return url_prefix
