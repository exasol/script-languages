import sys
import pandas as pd
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from io import StringIO


def parse_package_list_file(file_path:Path)->str:
    parse_package_list_file_command = ["perl","ext/scripts/list_newest_versions/extract_columns_from_package_lisl.pl","--file",str(file_path),"--columns","0,1"]
    result = subprocess.run(parse_package_list_file_command,stdout=subprocess.PIPE)
    result.check_returncode()
    result_string = result.stdout.decode("UTF-8")
    return result_string

def compare_package_lists(package_list_1:str, package_list_2:str)->pd.DataFrame:
    package_list_1_df = pd.read_csv(StringIO(package_list_1), delimiter="|", names=["package","version"])
    package_list_2_df = pd.read_csv(StringIO(package_list_2), delimiter="|", names=["package","version"])
    diff_df = pd.merge(package_list_1_df, package_list_2_df, how='outer', on='package',sort=False)
    diff_df = diff_df.sort_values("package")
    new = diff_df["version_y"].isnull()
    removed = diff_df["version_x"].isnull()
    updated =  ~diff_df["version_x"].isnull() & (diff_df["version_x"]!=diff_df["version_y"])
    diff_df["status"]=""
#    diff_df["status"].loc[new]="NEW"
#    diff_df["status",removed]="REMOVED"
    diff_df["status"].values[updated]="UPDATED"
    diff_df = diff_df.reset_index(drop=True)
    return diff_df

def compare_build_step(flavor_path:Path, build_step_path:Path, working_copy_1:Path, working_copy_2:Path):
    packages_path = Path(build_step_path,"packages")
    if packages_path.is_dir():
        for package_list_file  in packages_path.iterdir():
            package_list_working_copy_str_1 = parse_package_list_file(Path(working_copy_1,package_list_file))
            package_list_working_copy_str_2 = parse_package_list_file(Path(working_copy_2,package_list_file))
            diff = compare_package_lists(package_list_working_copy_str_1, package_list_working_copy_str_2)
            print(diff)


def compare_flavor(flavor_path:Path, working_copy_1:Path, working_copy_2:Path):
    flavor_base_path = Path(flavor_path,"flavor_base")
    if flavor_base_path.is_dir():
        for build_step_path in flavor_base_path.iterdir():
            if build_step_path.is_dir():
                compare_build_step(flavor_path,build_step_path,working_copy_1,working_copy_2)


get_last_tag_command = ["git","describe","--abbrev=0","--tags"]
last_tag_result = subprocess.run(get_last_tag_command,stdout=subprocess.PIPE)
last_tag_result.check_returncode()
last_tag = last_tag_result.stdout.decode("UTF-8").strip()
print(last_tag)

with TemporaryDirectory() as tmp_dir:
    print(tmp_dir)
    checkout_last_tag_command = ["git","worktree","add",tmp_dir,last_tag]
    checkout_last_tag_result = subprocess.run(checkout_last_tag_command,stderr=subprocess.PIPE)
    checkout_last_tag_result.check_returncode()
    init_submodule_command = ["git","submodule","update","--init"]
    init_submodule_result = subprocess.run(init_submodule_command,cwd=tmp_dir,stderr=subprocess.PIPE)
    init_submodule_result.check_returncode()
    working_copy_root = Path(".")
    for flavor_path in Path("flavors").iterdir():
        if flavor_path.is_dir():
            compare_flavor(flavor_path.relative_to(working_copy_root),working_copy_root,tmp_dir)


# package_list_1_path = sys.argv[1]
# package_list_2_path = sys.argv[2]
# package_list_1 = pd.read_csv(package_list_1_path, delimiter="|", names=["package","version"])
# package_list_2 = pd.read_csv(package_list_2_path, delimiter="|", names=["package","version"])
# diff = pd.merge(package_list_1, package_list_2, how='outer', on='package',sort=False)
# diff = diff.sort_values("package")
# diff["Updated"] = diff["version_x"]!=diff["version_y"]
# diff = diff.reset_index(drop=True)
# diff.to_markdown(sys.stdout)
# print()
