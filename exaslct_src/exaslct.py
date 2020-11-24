#! /usr/bin/env python3

from exasol_integration_test_docker_environment.cli.cli import cli
from exasol_integration_test_docker_environment.cli.commands import spawn_test_environment, push_test_container, \
    build_test_container

from exaslct_src.exaslct.cli.commands import build, clean_all_images, clean_flavor_images, export, \
    generate_language_activation, push, run_db_test, save, upload, clean

if __name__ == '__main__':
    # Required to announce the commands to click
    commands = [spawn_test_environment,
                push_test_container,
                build_test_container,
                build,
                clean_all_images,
                clean_flavor_images,
                export,
                generate_language_activation,
                push,
                run_db_test,
                save,
                upload,
                clean]
    cli()
