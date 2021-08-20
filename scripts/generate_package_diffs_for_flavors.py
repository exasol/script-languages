import sys
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from io import StringIO
import re
from typing import Dict, List
import tempfile

def parse_package_list_file(file_path:Path)->str:
    parse_package_list_file_command = [
            "perl","ext/scripts/list_newest_versions/extract_columns_from_package_lisl.pl",
            "--file",str(file_path),"--columns","0,1"]
    result = subprocess.run(parse_package_list_file_command,stdout=subprocess.PIPE)
    result.check_returncode()
    result_string = result.stdout.decode("UTF-8")
    return result_string

def compare_package_lists(package_list_1:str, package_list_2:str)->pd.DataFrame:
    package_list_1_df = pd.read_csv(StringIO(package_list_1), delimiter="|", names=["Package","Version1"])
    package_list_1_df["Version1"]=package_list_1_df["Version1"].replace("<<<<1>>>>","No version specified")
    package_list_2_df = pd.read_csv(StringIO(package_list_2), delimiter="|", names=["Package","Version2"])
    package_list_2_df["Version2"]=package_list_2_df["Version2"].replace("<<<<1>>>>","No version specified")
    diff_df = pd.merge(package_list_1_df, package_list_2_df, how='outer', on='Package',sort=False)
    new = diff_df["Version1"].isnull() & ~diff_df["Version2"].isnull()
    removed = diff_df["Version2"].isnull() & ~diff_df["Version1"].isnull()
    updated =  ~diff_df["Version1"].isnull() & ~diff_df["Version2"].isnull() & (diff_df["Version1"]!=diff_df["Version2"])
    diff_df["Status"]=""
    diff_df["Status"].values[new]="NEW"
    diff_df["Status"].values[removed]="REMOVED"
    diff_df["Status"].values[updated]="UPDATED"
    diff_df = diff_df.fillna("")
    diff_df = diff_df.sort_values("Status",ascending=False)
    diff_df = diff_df.reset_index(drop=True)
    return diff_df

def convert_requirements_file(package_list_str):
    def convert_line(line):
        line = line.replace("|<<<<1>>>>","|")
        line = line.replace("==","|")
        line = line.replace(">=","|")
        line = line.replace("<=","|")
        line = re.sub(r"\|$","",line)
        if not "|" in line:
            line += "|<<<<1>>>>"
        return line
    result = "\n".join(convert_line(line) for line in package_list_str.splitlines())
    return result

def load_package_file_or_alternative(working_copy:Path, package_list_file:Path, package_list_file_name_alternatives:Dict[str,List[str]]):
    package_list_str = ""
    possible_package_list_file_names = [package_list_file.name] 
    if package_list_file.name in package_list_file_name_alternatives:
        possible_package_list_file_names += package_list_file_name_alternatives[package_list_file.name]
    for package_list_file_name in possible_package_list_file_names:
        package_list_file = Path(package_list_file.parent, package_list_file_name)
        package_list_file_path = Path(working_copy, package_list_file)
        if package_list_file_path.exists():
            try:
                package_list_str = parse_package_list_file(Path(working_copy, package_list_file))
                if package_list_file_name in ["pip3_packages","pip_packages"]:
                    package_list_str = convert_requirements_file(package_list_str)
            except Exception as e:
                print(f"Could not parse {Path(working_copy, package_list_file_path)}")
                print(e)
            break
    return package_list_str

# TODO add second flavor parameter to compare also different flavors
def compare_build_step(flavor_path:Path, build_step_path:Path, working_copy_1:Path, working_copy_1_name:str, working_copy_2:Path, working_copy_2_name:str):
    package_list_file_name_alternatives = {
                "python3_pip_packages": ["pip3_packages"],
                "python2_pip_packages": ["pip_packages"]
            }
    result = {}
    packages_path = Path(build_step_path,"packages")
    if packages_path.is_dir():
        for package_list_file  in packages_path.iterdir():
            package_list_working_copy_str_1 = parse_package_list_file(Path(working_copy_1, package_list_file))
            package_list_working_copy_str_2 = load_package_file_or_alternative(working_copy_2, package_list_file, package_list_file_name_alternatives)
            diff = compare_package_lists(package_list_working_copy_str_2, package_list_working_copy_str_1)
            diff = diff.rename(columns={"Version1": f"Version in {working_copy_2_name}", "Version2": f"Version in {working_copy_1_name}"})
            result[package_list_file.name] = diff
    return result

# TODO add second flavor parameter to compare also different flavors
def compare_flavor(flavor_path:Path, working_copy_1:Path, working_copy_1_name:str, working_copy_2:Path, working_copy_2_name:str):
    flavor_base_path = Path(flavor_path,"flavor_base")
    result = {}
    if flavor_base_path.is_dir():
        for build_step_path in flavor_base_path.iterdir():
            if build_step_path.is_dir():
                diffs = compare_build_step(flavor_path, build_step_path, working_copy_1, working_copy_1_name, working_copy_2, working_copy_2_name)
                result[build_step_path.name]=diffs
    return result


def get_last_git_tag():
    get_last_tag_command = ["git","describe","--abbrev=0","--tags"]
    last_tag_result = subprocess.run(get_last_tag_command,stdout=subprocess.PIPE)
    last_tag_result.check_returncode()
    last_tag = last_tag_result.stdout.decode("UTF-8").strip()
    return last_tag

def checkout_git_tag_as_worktree(tmp_dir, last_tag):
    checkout_last_tag_command = ["git","worktree","add",tmp_dir,last_tag]
    checkout_last_tag_result = subprocess.run(checkout_last_tag_command,stderr=subprocess.PIPE)
    checkout_last_tag_result.check_returncode()
    init_submodule_command = ["git","submodule","update","--init"]
    init_submodule_result = subprocess.run(init_submodule_command,cwd=tmp_dir,stderr=subprocess.PIPE)
    init_submodule_result.check_returncode()


def get_package_list_diff_file_name(build_step, package_list):
    return f"{build_step}_{package_list}_diff.md"

def generate_dependency_diff_report_for_flavor(flavor_path_1:Path, flavor_path_2:Path, 
                                                working_copy_1_name:str, working_copy_2_name:str, 
                                                diffs, output_directory:Path):
    flavor1 = flavor_path_1.name.capitalize()
    flavor2 = flavor_path_2.name.capitalize()
    overview_page = f"# Package Version Comparison between {flavor1} flavor in {working_copy_1_name} and {flavor2} flavor in {working_copy_2_name}\n\n"
    for build_step in sorted(list(diffs.keys()),reverse=True):
        if len(diffs[build_step]) > 0:
            overview_page += f"- {build_step}\n" 
            for package_list in sorted(list(diffs[build_step].keys())):
                package_list_name = " ".join(word.capitalize() for word in package_list.split("_"))
                overview_page += f"  - [{package_list_name}]({get_package_list_diff_file_name(build_step,package_list)})\n"
            overview_file = Path(output_directory,"README.md")
            with overview_file.open("wt") as f:
                f.write(overview_page)


last_tag = get_last_git_tag()
output_directory = Path(tempfile.mkdtemp())
print(output_directory, output_directory)
with TemporaryDirectory() as tmp_dir:
    checkout_git_tag_as_worktree(tmp_dir, last_tag)
    working_copy_root = Path(".")
    working_copy_1_name = "HEAD"
    working_copy_2_name = last_tag
    for flavor_path in Path("flavors").iterdir():
        if flavor_path.is_dir():
            diffs = compare_flavor(flavor_path.relative_to(working_copy_root), working_copy_root, working_copy_1_name, tmp_dir, working_copy_2_name)
            generate_dependency_diff_report_for_flavor(flavor_path, flavor_path,
                                                        working_copy_1_name, working_copy_2_name,
                                                        diffs, output_directory)

