from typing import Tuple

from exaslct_src.test_environment.src.cli.cli import cli
from exaslct_src.test_environment.src.cli.common import set_build_config, set_docker_repository_config, run_task, add_options, \
    import_build_steps, set_job_id
from exaslct_src.exaslct.cli.options.flavor_options import flavor_options
from exaslct_src.exaslct.cli.options.goal_options import goal_options
from exaslct_src.exaslct.lib.tasks.push.docker_push import DockerFlavorsPush
from exaslct_src.test_environment.src.cli.options.build_options import build_options
from exaslct_src.test_environment.src.cli.options.docker_repository_options import docker_repository_options
from exaslct_src.test_environment.src.cli.options.push_options import push_options
from exaslct_src.test_environment.src.cli.options.system_options import system_options


@cli.command()
@add_options(flavor_options)
@add_options(goal_options)
@add_options(push_options)
@add_options(build_options)
@add_options(docker_repository_options)
@add_options(system_options)
def push(flavor_path: Tuple[str, ...],
         goal: Tuple[str, ...],
         force_push: bool,
         push_all: bool,
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
    set_job_id(DockerFlavorsPush.__name__)
    task_creator = lambda: DockerFlavorsPush(force_push=force_push,
                                             push_all=push_all,
                                             goals=list(goal),
                                             flavor_paths=list(flavor_path))
    success, task = run_task(task_creator, workers, task_dependencies_dot_file)
    if not success:
        exit(1)
