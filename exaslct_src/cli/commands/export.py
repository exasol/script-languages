from typing import Tuple

import luigi
from click._unicodefun import click

from exaslct_src import ExportContainer
from exaslct_src.cli.cli import cli
from exaslct_src.cli.common import set_build_config, set_docker_config, run_tasks, add_options
from exaslct_src.cli.options \
    import build_options, flavor_options, system_options, release_options, \
    docker_options_login_not_required


@cli.command()
@add_options(flavor_options)
@add_options(release_options)
@click.option('--output-path', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None)
@click.option('--release-name', type=str, default=None)
@add_options(build_options)
@add_options(docker_options_login_not_required)
@add_options(system_options)
def export(flavor_path: Tuple[str, ...],
           release_type: str,
           output_path: str,
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
    This command exports the whole script language container package of the flavor,
    ready for the upload into the bucketfs. If the stages do not exists locally,
    the system will build or pull them before the exporting the packaged container.
    """
    set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)
    tasks = [ExportContainer(flavor_paths=list(flavor_path),
                             release_types=list([release_type]),
                             output_path=output_path,
                             release_name=release_name
                             )]

    def on_success():
        target = luigi.LocalTarget(
            "%s/exports/current" % (output_directory))

        with target.open("r") as f:
            print(f.read())

    run_tasks(tasks, workers, on_success=on_success)
