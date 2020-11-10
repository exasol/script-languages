from typing import Tuple

from exasol_integration_test_docker_environment.cli.cli import cli
from exasol_integration_test_docker_environment.cli.common import add_options, set_output_directory, \
    set_docker_repository_config, set_job_id, run_task, import_build_steps
from exasol_integration_test_docker_environment.cli.options.docker_repository_options import \
    simple_docker_repository_options
from exasol_integration_test_docker_environment.cli.options.system_options import output_directory_option, \
    system_options

from exaslct_src.exaslct.cli.options.flavor_options import flavor_options
from exaslct_src.exaslct.lib.tasks.clean.clean_images import CleanExaslcAllImages, CleanExaslcFlavorsImages


@cli.command()
@add_options(flavor_options)
@add_options([output_directory_option])
@add_options(simple_docker_repository_options)
@add_options(system_options)
def clean_flavor_images(flavor_path: Tuple[str, ...],
                        output_directory: str,
                        docker_repository_name: str,
                        docker_tag_prefix: str,
                        workers: int,
                        task_dependencies_dot_file: str):
    """
    This command uploads the whole script language container package of the flavor to the database.
    If the stages or the packaged container do not exists locally, the system will build, pull or
    export them before the upload.
    """
    import_build_steps(flavor_path)
    set_output_directory(output_directory)
    set_docker_repository_config(None, docker_repository_name, None, docker_tag_prefix, "source")
    set_docker_repository_config(None, docker_repository_name, None, docker_tag_prefix, "target")
    set_job_id(CleanExaslcFlavorsImages.__name__)
    task_creator = lambda: CleanExaslcFlavorsImages(flavor_paths=list(flavor_path))
    success, task = run_task(task_creator, workers, task_dependencies_dot_file)
    if not success:
        exit(1)


@cli.command()
@add_options([output_directory_option])
@add_options(simple_docker_repository_options)
@add_options(system_options)
def clean_all_images(
        output_directory: str,
        docker_repository_name: str,
        docker_tag_prefix: str,
        workers: int,
        task_dependencies_dot_file: str):
    """
    This command uploads the whole script language container package of the flavor to the database.
    If the stages or the packaged container do not exists locally, the system will build, pull or
    export them before the upload.
    """
    set_output_directory(output_directory)
    set_docker_repository_config(None, docker_repository_name, None, docker_tag_prefix, "source")
    set_docker_repository_config(None, docker_repository_name, None, docker_tag_prefix, "target")
    set_job_id(CleanExaslcAllImages.__name__)
    task_creator = lambda: CleanExaslcAllImages()
    success, task = run_task(task_creator, workers, task_dependencies_dot_file)
    if not success:
        exit(1)

# TODO add commands clean containers, networks, all
