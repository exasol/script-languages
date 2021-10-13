import re
import subprocess
import tempfile
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

import click
import pandas as pd


def parse_package_list_file(file_path: Path) -> str:
    parse_package_list_file_command = [
        "perl", "ext/scripts/list_newest_versions/extract_columns_from_package_lisl.pl",
        "--file", str(file_path), "--columns", "0,1"]
    result = subprocess.run(parse_package_list_file_command, stdout=subprocess.PIPE)
    result.check_returncode()
    result_string = result.stdout.decode("UTF-8")
    return result_string


def compare_package_lists(package_list_1: str, package_list_2: str) -> pd.DataFrame:
    package_list_1_df = pd.read_csv(StringIO(package_list_1), delimiter="|", names=["Package", "Version1"])
    package_list_1_df["Version1"] = package_list_1_df["Version1"].replace("<<<<1>>>>", "No version specified")
    package_list_2_df = pd.read_csv(StringIO(package_list_2), delimiter="|", names=["Package", "Version2"])
    package_list_2_df["Version2"] = package_list_2_df["Version2"].replace("<<<<1>>>>", "No version specified")
    diff_df = pd.merge(package_list_1_df, package_list_2_df, how='outer', on='Package', sort=False)
    new = diff_df["Version1"].isnull() & ~diff_df["Version2"].isnull()
    removed = diff_df["Version2"].isnull() & ~diff_df["Version1"].isnull()
    updated = ~diff_df["Version1"].isnull() & ~diff_df["Version2"].isnull() & (
            diff_df["Version1"] != diff_df["Version2"])
    diff_df["Status"] = ""
    diff_df["Status"].values[new] = "NEW"
    diff_df["Status"].values[removed] = "REMOVED"
    diff_df["Status"].values[updated] = "UPDATED"
    diff_df = diff_df.fillna("")
    diff_df = diff_df.sort_values("Status", ascending=False)
    diff_df = diff_df.reset_index(drop=True)
    return diff_df


def convert_requirements_file(package_list_str: str) -> str:
    def convert_line(line):
        line = line.replace("|<<<<1>>>>", "|")
        line = line.replace("==", "|")
        line = line.replace(">=", "|")
        line = line.replace("<=", "|")
        line = re.sub(r"\|$", "", line)
        if "|" not in line:
            line += "|<<<<1>>>>"
        return line

    result = "\n".join(convert_line(line) for line in package_list_str.splitlines())
    return result


def find_package_file_or_alternative(working_copy: Path,
                                     build_step_path: Path,
                                     package_list_file_name: str,
                                     package_list_file_name_alternatives: Dict[str, List[str]]) -> Optional[str]:
    possible_package_list_file_names = [package_list_file_name]
    if package_list_file_name in package_list_file_name_alternatives:
        possible_package_list_file_names += package_list_file_name_alternatives[package_list_file_name]
    packages_directory = Path(working_copy, build_step_path, "packages")
    if packages_directory.exists():
        for package_list_file_name in possible_package_list_file_names:
            package_list_file_path = Path(packages_directory, package_list_file_name)
            if package_list_file_path.exists():
                return package_list_file_name
    return None


def load_package_file_or_alternative(working_copy: Path,
                                     package_list_file: Path):
    package_list_str = ""
    try:
        package_list_str = parse_package_list_file(Path(working_copy, package_list_file))
        if package_list_file.name in ["pip3_packages", "pip_packages"]:
            package_list_str = convert_requirements_file(package_list_str)
    except Exception as e:
        print(f"Could not parse {Path(working_copy, package_list_file)}")
        print(e)
    return package_list_str


def compare_build_step(build_step_path_1: Path, working_copy_1: Path, working_copy_1_name: str,
                       build_step_path_2: Path, working_copy_2: Path, working_copy_2_name: str) \
        -> Dict[Tuple[str, Optional[str]], pd.DataFrame]:
    package_list_file_name_alternatives = {
        "python3_pip_packages": ["pip3_packages"],
        "python2_pip_packages": ["pip_packages"]
    }
    result = {}
    packages_path_1 = Path(build_step_path_1, "packages")
    if packages_path_1.is_dir():
        for package_list_file_1 in packages_path_1.iterdir():
            package_list_file_name_1 = package_list_file_1.name
            package_list_working_copy_str_1 = parse_package_list_file(Path(working_copy_1, package_list_file_1))
            package_list_file_name_2 = find_package_file_or_alternative(working_copy_2,
                                                                        build_step_path_2,
                                                                        package_list_file_name_1,
                                                                        package_list_file_name_alternatives)
            result_key = (package_list_file_name_1, package_list_file_name_2)
            if package_list_file_name_2 is None:
                package_list_working_copy_str_2 = ""
            else:
                package_list_file_2 = Path(build_step_path_2, "packages", package_list_file_name_2)
                package_list_working_copy_str_2 = load_package_file_or_alternative(working_copy_2,
                                                                                   package_list_file_2)
            diff_df = compare_package_lists(package_list_working_copy_str_2, package_list_working_copy_str_1)
            new_version1_name = f"Version in {working_copy_2_name}"
            new_version2_name = f"Version in {working_copy_1_name}"

            diff_df = diff_df.rename(columns={"Version1": new_version1_name,
                                              "Version2": new_version2_name})
            diff_df = diff_df[["Package",new_version1_name,new_version2_name,"Status"]]
            result[result_key] = diff_df
    return result


def compare_flavor(flavor_path_1: Path, working_copy_1: Path, working_copy_1_name: str,
                   flavor_path_2: Path, working_copy_2: Path, working_copy_2_name: str) \
        -> Dict[Tuple[str, str], Dict[Tuple[str, Optional[str]], pd.DataFrame]]:
    flavor_base_path_1 = Path(flavor_path_1, "flavor_base")
    flavor_base_path_2 = Path(flavor_path_2, "flavor_base")
    result = {}
    if flavor_base_path_1.is_dir():
        for build_step_path_1 in flavor_base_path_1.iterdir():
            if build_step_path_1.is_dir():
                build_step_name_1 = build_step_path_1.name
                build_step_name_2 = build_step_name_1
                build_step_path_2 = Path(flavor_base_path_2, build_step_name_2)
                diffs = compare_build_step(build_step_path_1, working_copy_1, working_copy_1_name,
                                           build_step_path_2, working_copy_2, working_copy_2_name)
                result[(build_step_name_1, build_step_name_2)] = diffs
    return result


def get_last_git_tag() -> str:
    get_fetch_command = ["git", "fetch"]
    fetch_result = subprocess.run(get_fetch_command, stderr=subprocess.PIPE)
    fetch_result.check_returncode()
    get_last_tag_command = ["git", "describe", "--abbrev=0", "--tags", "origin/master"]
    last_tag_result = subprocess.run(get_last_tag_command, stdout=subprocess.PIPE)
    last_tag_result.check_returncode()
    last_tag = last_tag_result.stdout.decode("UTF-8").strip()
    return last_tag


def checkout_git_tag_as_worktree(tmp_dir, last_tag):
    checkout_last_tag_command = ["git", "worktree", "add", tmp_dir, last_tag]
    checkout_last_tag_result = subprocess.run(checkout_last_tag_command, stderr=subprocess.PIPE)
    checkout_last_tag_result.check_returncode()
    init_submodule_command = ["git", "submodule", "update", "--init"]
    init_submodule_result = subprocess.run(init_submodule_command, cwd=tmp_dir, stderr=subprocess.PIPE)
    init_submodule_result.check_returncode()


def generate_dependency_diff_report_for_package_list(
        package_file_diff_file: Path,
        diff_df: pd.DataFrame):
    package_file_diff_file.parent.mkdir(parents=True, exist_ok=True)
    with package_file_diff_file.open("wt") as f:
        f.write("<!-- markdown-link-check-disable -->\n\n")
        diff_df.to_markdown(f)


def generate_dependency_diff_report_for_build_step(
        build_steps: Tuple[str, str],
        diffs: Dict[Tuple[str, Optional[str]], pd.DataFrame],
        base_output_directory: Path,
        relative_output_directory: Path):
    result = ""
    if len(diffs) > 0:
        build_step_caption = generate_build_step_caption(build_steps)
        result = f"- {build_step_caption}\n"
        for package_lists in sorted(list(diffs.keys())):
            package_list_caption = \
                generate_package_list_caption(package_lists)
            relative_package_file_diff_file = Path(relative_output_directory, f"{package_lists[0]}_diff.md")
            result += f"  - [{package_list_caption}]({relative_package_file_diff_file})\n"
            package_file_diff_file = Path(base_output_directory, relative_package_file_diff_file)
            generate_dependency_diff_report_for_package_list(
                package_file_diff_file, diffs[package_lists])
    return result


def generate_build_step_caption(build_steps):
    build_step_1_capitalized = build_steps[0].capitalize()
    if build_steps[1] is None or build_steps[0] == build_steps[1]:
        build_step_caption = f"Comparison of build step {build_step_1_capitalized}"
    else:
        build_step_2_capitalized = build_steps[1].capitalize()
        build_step_caption = f"Comparison of build steps {build_step_1_capitalized} and {build_step_2_capitalized}"
    return build_step_caption


def generate_package_list_caption(package_lists: Tuple[str, Optional[str]], ) -> str:
    package_list_name_1 = " ".join(word.capitalize() for word in package_lists[0].split("_"))
    if package_lists[1] is None or package_lists[0] == package_lists[1]:
        if package_lists[0] == package_lists[1]:
            package_list_caption = f"Comparison of package list {package_list_name_1}"
        else:
            package_list_caption = f"New package list {package_list_name_1}"
    else:
        package_list_name_2 = " ".join(word.capitalize() for word in package_lists[1].split("_"))
        package_list_caption = f"Comparison of package lists {package_list_name_1} and {package_list_name_2}"
    return package_list_caption


def generate_dependency_diff_report_for_flavor(flavor_name_1: str, working_copy_1_name: str,
                                               flavor_name_2: str, working_copy_2_name: str,
                                               diffs: Dict[
                                                   Tuple[str, str], Dict[Tuple[str, Optional[str]], pd.DataFrame]],
                                               base_output_directory: Path, relative_output_directory: Path):
    relative_overview_file = Path(relative_output_directory, "README.md")
    overview_file = Path(base_output_directory, relative_overview_file)
    overview_file.parent.mkdir(parents=True, exist_ok=True)
    flavor_name_1_capitalized = flavor_name_1.capitalize()
    flavor_name_2_capitalized = flavor_name_2.capitalize()
    overview_file_content = \
        f"# Package Version Comparison between " \
        f"{flavor_name_1_capitalized} flavor in {working_copy_1_name} and " \
        f"{flavor_name_2_capitalized} flavor in {working_copy_2_name}\n\n"
    if flavor_name_1 == flavor_name_1:
        result = f"- [Comparison of flavor {flavor_name_1_capitalized}" \
                 f"]({relative_overview_file})\n"
    else:
        result = f"- [Comparison of flavors " \
                 f"{flavor_name_1_capitalized} and {flavor_name_2_capitalized}" \
                 f"]({relative_overview_file})\n"
    for build_steps in sorted(list(diffs.keys()), reverse=True):
        build_step_base_output_directory = Path(base_output_directory, relative_output_directory)
        build_step_relative_output_directory = Path(build_steps[0])
        overview_file_content += \
            generate_dependency_diff_report_for_build_step(
                build_steps,
                diffs[build_steps],
                build_step_base_output_directory,
                build_step_relative_output_directory)
    with overview_file.open("wt") as f:
        f.write(overview_file_content)
    return result


def generate_dependency_diff_report_for_all_flavors(working_copy_1_root: Path,
                                                    working_copy_1_name: str,
                                                    working_copy_2_root: Path,
                                                    working_copy_2_name: str,
                                                    base_output_directory: Path):
    base_output_directory.mkdir(parents=True, exist_ok=True)
    overview_file = Path(base_output_directory, "README.md")
    overview_file_content = \
        f"# Package Version Comparison between " \
        f"{working_copy_1_name} and " \
        f"{working_copy_2_name}\n\n"
    for flavor_path in Path(working_copy_1_root, "flavors").iterdir():
        if flavor_path.is_dir():
            relative_flavor_path = flavor_path.relative_to(working_copy_1_root)
            diffs = compare_flavor(relative_flavor_path, working_copy_1_root, working_copy_1_name,
                                   relative_flavor_path, working_copy_2_root, working_copy_2_name)
            if len(diffs) > 0:
                flavor_1 = flavor_path.name
                flavor_2 = flavor_path.name
                if flavor_1 == flavor_2:
                    flavor_relative_output_directory = Path(flavor_1)
                else:
                    flavor_relative_output_directory = Path(f"{flavor_1}__{flavor_2}")
                overview_file_content += \
                    generate_dependency_diff_report_for_flavor(flavor_1, working_copy_1_name,
                                                               flavor_2, working_copy_2_name,
                                                               diffs,
                                                               base_output_directory,
                                                               flavor_relative_output_directory)
    with overview_file.open("wt") as f:
        f.write(overview_file_content)

@click.command()
@click.option('--output-directory', required=True, help="Directory where the diff reports are generated",
              type=click.Path(exists=False))
@click.option('--current-working-copy-name', required=True, help="Name of the current git working copy. "
                                                                 "For example, the version of a new release.",
              type=str)
def main(output_directory:str, current_working_copy_name:str):
    last_tag = get_last_git_tag()
    with TemporaryDirectory() as working_copy_2_root:
        checkout_git_tag_as_worktree(working_copy_2_root, last_tag)
        working_copy_root = Path(".")
        working_copy_1_name = current_working_copy_name
        working_copy_2_name = last_tag
        generate_dependency_diff_report_for_all_flavors(working_copy_root, working_copy_1_name,
                                                        working_copy_2_root, working_copy_2_name,
                                                        Path(output_directory))


if __name__ == '__main__':
    main()
