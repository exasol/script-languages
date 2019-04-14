from typing import Tuple

import luigi
from click._unicodefun import click

from exaslct_src import UploadContainer
from exaslct_src.cli.cli import cli
from exaslct_src.cli.common import set_build_config, set_docker_config, run_tasks, add_options
from exaslct_src.cli.options \
    import build_options, flavor_options, docker_options, system_options, release_options


@cli.command()
@add_options(flavor_options)
@add_options(release_options)
@click.option('--database-host', type=str,
              required=True)
@click.option('--bucketfs-port', type=int, required=True)
@click.option('--bucketfs-username', type=str, required=True)
@click.option('--bucketfs-password', type=str)
@click.option('--bucketfs-https/--no-bucketfs-https', default=False)
@click.option('--bucketfs-name', type=str, required=True)
@click.option('--bucket-name', type=str, required=True)
@click.option('--path-in-bucket', type=str, required=True)
@click.option('--release-name', type=str, default=None)
@add_options(build_options)
@add_options(docker_options)
@add_options(system_options)
def upload(flavor_path: Tuple[str, ...],
           release_type: str,
           database_host: str,
           bucketfs_port: int,
           bucketfs_username: str,
           bucketfs_password: str,
           bucketfs_https: bool,
           bucketfs_name: str,
           bucket_name: str,
           path_in_bucket: str,
           release_name: str,
           force_build: bool,
           force_pull: bool,
           output_directory: str,
           temporary_base_directory: str,
           log_build_context_content: bool,
           docker_base_url: str,
           docker_repository_name: str,
           docker_username: str,
           docker_password: str,
           workers: int):
    """
    This command uploads the whole script language container package of the flavor to the database.
    If the stages or the packaged container do not exists locally, the system will build, pull or
    export them before the upload.
    """
    set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)
    tasks = [UploadContainer(flavor_paths=list(flavor_path),
                             release_types=list([release_type]),
                             database_host=database_host,
                             bucketfs_port=bucketfs_port,
                             bucketfs_username=bucketfs_username,
                             bucketfs_password=bucketfs_password,
                             bucket_name=bucket_name,
                             path_in_bucket=path_in_bucket,
                             bucketfs_https=bucketfs_https,
                             release_name=release_name,
                             bucketfs_name=bucketfs_name
                             )]

    def on_success():
        target = luigi.LocalTarget(
            "%s/uploads/current" % (output_directory))

        with target.open("r") as f:
            print(f.read())

    run_tasks(tasks, workers, on_success=on_success)

