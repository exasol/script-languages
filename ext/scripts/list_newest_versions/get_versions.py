import json
from  urllib.request import urlopen
import sys
from distutils.version import LooseVersion, StrictVersion
from typing import List

PRERELEASE_KEYWORDS=["alpha","beta","rc"]

def is_version_part_prerelease(index:int, version_part:str):
    if version_part.isnumeric():
        return False
    if any([keyword in version_part.lower() for keyword in PRERELEASE_KEYWORDS]):
        return True
    if index>1 and ("a" in version_part or "b" in version_part):
        return True

def is_version_prerelease(version_parts:List[str]):
    return any([is_version_part_prerelease(index, part) for index,part in enumerate(version_parts)])

def split_version(version:str):
    version_parts = version.split(".")
    return version_parts

def fetch_versions(package_name:str, current_version:str):
    url = "https://pypi.org/pypi/%s/json" % (package_name,)
    data = json.load(urlopen(url))
    try:
        versions = list(data["releases"].keys())
        versions.sort(key=StrictVersion)
    except:
        try:
            versions = list(data["releases"].keys())
            versions.sort(key=LooseVersion)
        except:
            versions = list(data["releases"].keys())
            versions.sort(key=lambda x: split_version(x))
    index=versions.index(current_version)
    return versions[index:]

def join_version(version_parts:List[str]):
    return ".".join(version_parts)

def get_newest_version_by_predicate(
        versions:List[str], current_version:str,
        current_version_predicate, newest_version_predicate):
    current_version_parts = split_version(current_version)
    versions_splitted = [split_version(version) for version in versions]
    versions_without_prereleases = [version for version in versions_splitted if not is_version_prerelease(version)]
    if current_version_predicate(current_version_parts):
        newest_version = None
        for version_parts in versions_without_prereleases:
            if newest_version_predicate(current_version_parts, version_parts):
                newest_version = version_parts
            else:
                return join_version(newest_version)
        if newest_version is not None:
            return join_version(newest_version)
        else:
            return current_version
    else:
        return current_version

def get_newest_bugfix_version(versions:List[str], current_version:str):
    current_version_predicate = lambda current_version_parts: len(current_version_parts) > 2
    newest_version_predicate = lambda current_version_parts, version_parts: len(version_parts) > 2 \
                    and version_parts[0] == current_version_parts[0] \
                    and version_parts[1] == current_version_parts[1]
    return get_newest_version_by_predicate(
            versions, current_version,
            current_version_predicate, newest_version_predicate)


def get_newest_minor_version(versions:List[str], current_version:str):
    current_version_predicate = lambda current_version_parts: len(current_version_parts) > 1
    newest_version_predicate = lambda current_version_parts, version_parts: len(version_parts) > 1 \
                    and version_parts[0] == current_version_parts[0]
    return get_newest_version_by_predicate(
            versions, current_version,
            current_version_predicate, newest_version_predicate)

def get_newest_major_version(versions:List[str], current_version:str):
    current_version_predicate = lambda current_version_parts: len(current_version_parts) > 0
    newest_version_predicate = lambda current_version_parts, version_parts: len(version_parts) > 0
    return get_newest_version_by_predicate(
            versions, current_version,
            current_version_predicate, newest_version_predicate)

try:
    package_info = sys.argv[1]
    package_name, current_version = package_info.split("|")
    versions = fetch_versions(package_name, current_version)
    new_versions = ";".join(versions)
    newest_bugfix_version = get_newest_bugfix_version(versions, current_version)
    newest_minor_version = get_newest_minor_version(versions, current_version)
    newest_major_version = get_newest_major_version(versions, current_version)
    print(f"{package_name}|{current_version}|{newest_bugfix_version}|{newest_minor_version}|{newest_major_version}|{new_versions}")
except Exception as e:
    print(f"{package_name}|{current_version}")
    print(f"Error looking up {package_name}: {e}",file=sys.stderr)

