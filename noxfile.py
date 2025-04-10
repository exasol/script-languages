import argparse
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import nox
from exasol.slc.api import run_db_tests as exaslct_run_db_tests

ROOT = Path(__file__).parent
FLAVOR_PATH = ROOT / "flavors"

# default actions to be run if nothing is explicitly specified with the -s option
nox.options.sessions = []

def get_flavors() -> List[Path]:
    flavor_names = [f.name for f in FLAVOR_PATH.iterdir() if f.is_dir()]
    return flavor_names

def get_oft_jar(session: nox.Session) -> Path:
    oft_version = "4.1.0"
    oft_jar = Path.home() / ".m2" / "repository" / "org" / "itsallcode" / "openfasttrace" / "openfasttrace" / oft_version / f"openfasttrace-{oft_version}.jar"
    if not oft_jar.exists():
        print(f"Downloading OpenFastTrace {oft_version}")
        session.run("mvn", "--batch-mode", "org.apache.maven.plugins:maven-dependency-plugin:3.3.0:get", f"-Dartifact=org.itsallcode.openfasttrace:openfasttrace:{oft_version}")
    return oft_jar

def run_oft_for_udf_client(session: nox.Session, *args) -> None:
    oft_jar = get_oft_jar(session)
    udf_client_base_dir = ROOT / "exaudfclient"
    udf_client_src_dir = udf_client_base_dir / "base"

    with session.chdir(ROOT):
        session.run(
            "java",
            "-jar",
            oft_jar,
            "trace",
            "-a",
            "feat,req,dsn",
            f"{udf_client_base_dir}/docs",
            f"{udf_client_src_dir}",
            "-t",
            "V2,_",
            *args
        )


@nox.session(name="run-oft", python=False)
def run_oft_udf_client_plaintext(session: nox.Session):
    """
    Downloads (if needed) OFT and executes it for the udf client for tag "V2,_" printing the output to stdout.
    """
    run_oft_for_udf_client(session)


@nox.session(name="run-oft-html", python=False)
def run_oft_udf_client_html(session: nox.Session):
    """
    Downloads (if needed) OFT and executes it for the udf client for tag "V2,_" creating a html page as output.
    """
    html_file = session.posargs[0] if session.posargs else "report.html"
    run_oft_for_udf_client(session, "-o", "html", "-f", html_file)

@nox.session(name="get-flavors", python=False)
def run_get_flavors(session: nox.Session):
    """
    Print all flavors as JSON.
    """
    print(json.dumps(get_flavors()))
    #print(json.dumps(["template-Exasol-all-python-3.10"]))

@nox.session(name="get-build-runner-for-flavor", python=False)
@nox.parametrize("flavor", get_flavors())
def run_get_build_runner_for_flavor(session: nox.Session, flavor: str):
    """
    Returns the runner for a flavor
    """
    ci_file = FLAVOR_PATH / flavor / "ci.json"
    runner = "ubuntu-22.04"
    if ci_file.exists():
        with open(ci_file) as file:
            ci = json.load(file)
            runner = ci["build_runner"]
    print(runner)

@nox.session(name="get-test-runner-for-flavor", python=False)
@nox.parametrize("flavor", get_flavors())
def run_test_get_runner_for_flavor(session: nox.Session, flavor: str):
    """
    Returns the test-runner for a flavor
    """
    ci_file = FLAVOR_PATH / flavor / "ci.json"
    runner = "ubuntu-22.04"
    if ci_file.exists():
        with open(ci_file) as file:
            ci = json.load(file)
            runner = ci["test_config"]["test_runner"]
    print(runner)

@nox.session(name="get-test-get-names-for-flavor", python=False)
@nox.parametrize("flavor", get_flavors())
def run_test_get_names_for_flavor(session: nox.Session, flavor: str):
    """
    Returns the test-runner for a flavor
    """
    ci_file = FLAVOR_PATH / flavor / "ci.json"
    test_sets_names = []
    if ci_file.exists():
        with open(ci_file) as file:
            ci = json.load(file)
            test_sets = ci["test_config"]["test_sets"]
            test_sets_names = [test_set["name"] for test_set in test_sets]
    print(json.dumps(test_sets_names))

@nox.session(name="run-db-tests", python=True)
def run_db_tests(session: nox.Session):
    """
    Returns the test-runner for a flavor
    """
    def parser() -> ArgumentParser:
        p = ArgumentParser(
            usage="nox -s get-test-set-folders -- --flavor <flavor> --test-set <test-set-name>",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p.add_argument("--flavor")
        p.add_argument("--test-set-name")
        return p

    args = parser().parse_args(session.posargs)
    ci_file = FLAVOR_PATH / args.flavor / "ci.json"
    test_set_folders = []
    if ci_file.exists():
        with open(ci_file) as file:
            ci = json.load(file)
            test_set = ci["test_config"]["test_sets"][args.test_set_name]
            test_set_folders=(folder for folder in test_set["folders"])
    exaslct_run_db_tests.run_db_test(flavor_path=f"flavors/{args.flavor}", test_set_folders=test_set_folders)
