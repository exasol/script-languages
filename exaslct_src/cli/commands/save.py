from typing import Tuple

from click._unicodefun import click

from exaslct_src.cli.cli import cli
from exaslct_src.cli.common import set_build_config, set_docker_repository_config, run_task, add_options, \
    import_build_steps, set_job_id
from exaslct_src.cli.options \
    import build_options, flavor_options, system_options, goal_options, \
    docker_repository_options
from exaslct_src.lib.docker_save import DockerSave


@cli.command()
@click.option('--save-directory',
              type=click.Path(file_okay=False, dir_okay=True),
              help="Directory where to save the image tarballs")
@click.option('--force-save/--no-force-save', default=False,
              help="Forces the system to overwrite existing save for build steps that run")
@click.option('--save-all/--no-save-all', default=False,
              help="Forces the system to save all images of build-steps that are specified by the goals")
@add_options(flavor_options)
@add_options(goal_options)
@add_options(build_options)
@add_options(docker_repository_options)
@add_options(system_options)
def save(save_directory: str,
         force_save: bool,
         save_all: bool,
         flavor_path: Tuple[str, ...],
         goal: Tuple[str, ...],
         force_rebuild: bool,
         force_rebuild_from: Tuple[str, ...],
         force_pull: bool,
         output_directory: str,
         temporary_base_directory: str,
         log_build_context_content: bool,
         cache_directory: str,
         build_name: str,
         source_docker_repository_name: str,
         source_docker_tag_prefix: str,
         source_docker_username: str,
         source_docker_password: str,
         target_docker_repository_name: str,
         target_docker_tag_prefix: str,
         target_docker_username: str,
         target_docker_password: str,
         workers: int,
         task_dependencies_dot_file: str):
    """
    This command pushes all stages of the script language container flavor.
    If the stages do not exists locally, the system will build or pull them before the push.
    """
    import_build_steps(flavor_path)
    set_build_config(force_rebuild,
                     force_rebuild_from,
                     force_pull,
                     log_build_context_content,
                     output_directory,
                     temporary_base_directory,
                     cache_directory,
                     build_name)
    set_docker_repository_config(source_docker_password, source_docker_repository_name, source_docker_username,
                                 source_docker_tag_prefix, "source")
    set_docker_repository_config(target_docker_password, target_docker_repository_name, target_docker_username,
                                 target_docker_tag_prefix, "target")
    set_job_id(DockerSave.__name__)
    task_creator = lambda: DockerSave(save_path=save_directory,
                                      force_save=force_save,
                                      save_all=save_all,
                                      flavor_paths=list(flavor_path),
                                      goals=list(goal))
    success, task = run_task(task_creator, workers, task_dependencies_dot_file)
    if not success:
        exit(1)
