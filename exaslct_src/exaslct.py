#! /usr/bin/env python3
#
# noinspection PyUnresolvedReferences
from exaslct_src.exaslct.cli.commands import build, clean_all_images, clean_flavor_images, export, \
    generate_language_activation, push, run_db_test, save, upload, clean
from exaslct_src.test_environment.src.cli.cli import cli
# noinspection PyUnresolvedReferences
from exaslct_src.test_environment.src.cli.commands import spawn_test_environment, push_test_container, build_test_container

if __name__ == '__main__':
    cli()
