from typing import Tuple

from exaslct_src import CleanExaslcImages
from exaslct_src.cli.cli import cli
from exaslct_src.cli.common import set_docker_config, run_tasks, set_output_directory, add_options
from exaslct_src.cli.options \
    import flavor_options, system_options, output_directory, docker_options_login_not_required


@cli.command()
@add_options(flavor_options)
@add_options([output_directory])
@add_options(docker_options_login_not_required)
@add_options(system_options)
def clean_flavor_images(flavor_path: Tuple[str, ...],
                        output_directory: str,
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
    set_output_directory(output_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)
    tasks = lambda: [CleanExaslcImages(flavor_path=flavor_path[0])]
    run_tasks(tasks, workers)


@cli.command()
@add_options([output_directory])
@add_options(docker_options_login_not_required)
@add_options(system_options)
def clean_all_images(
        output_directory: str,
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
    set_output_directory(output_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)
    tasks = lambda: [CleanExaslcImages()]
    run_tasks(tasks, workers)

# TODO add commands clean containers, networks, all
