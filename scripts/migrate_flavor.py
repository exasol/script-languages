from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from exasol.exaslpm.model.package_file_config import (
    AptPackage,
    AptPackages,
    BuildStep,
    CondaBinary,
    CondaPackage,
    CondaPackages,
    Micromamba,
    PackageFile,
    Phase,
    PipPackage,
    PipPackages,
    RPackage,
    RPackages,
    Tools,
    ValidationConfig,
)
from exasol.exaslpm.model.serialization import to_yaml_str

FLAVORS_DIR = Path("flavors")
PUBLIC_PACKAGE_FILE = Path("packages.yml")
INTERNAL_PACKAGE_FILE = Path("flavor_base") / "packages.yml"

BUILD_STEP_MAPPING = {
    "base_test_build_run": INTERNAL_PACKAGE_FILE,
    "base_test_deps": INTERNAL_PACKAGE_FILE,
    "build_deps": PUBLIC_PACKAGE_FILE,
    "build_run": PUBLIC_PACKAGE_FILE,
    "flavor_base_deps": PUBLIC_PACKAGE_FILE,
    "flavor_test_build_run": INTERNAL_PACKAGE_FILE,
    "language_deps": PUBLIC_PACKAGE_FILE,
    "release": PUBLIC_PACKAGE_FILE,
    "security_scan": INTERNAL_PACKAGE_FILE,
    "udfclient_deps": PUBLIC_PACKAGE_FILE,
    "conda_deps": PUBLIC_PACKAGE_FILE,
    "conda": PUBLIC_PACKAGE_FILE,
    "flavor_customization": PUBLIC_PACKAGE_FILE,
}

BUILD_STEP_ALIASES = {
    "flavor_base_deps_apt": "flavor_base_deps",
    "flavor_base_deps_python": "flavor_base_deps",
    "flavor_base_deps_r": "flavor_base_deps",
}

PHASE_NAME_BY_INSTALLER = {
    "apt": "install_apt_packages",
    "pip": "install_pip_packages",
    "r": "install_r_packages",
    "conda": "install_conda_packages",
}

CONDA_BINARY_MAPPING = {
    "$MAMBA_ROOT_PREFIX/bin/mamba": CondaBinary.Mamba,
    "/bin/micromamba": CondaBinary.Micromamba,
    "$MAMBA_ROOT_PREFIX/bin/conda": CondaBinary.Conda,
}


def _parse_legacy_line(line: str) -> tuple[str | None, str | None]:
    """
    Parse a package definition line.
    Inline comments are only considered if '#' is preceded by whitespace,
    so URLs like '...#egg=...' remain intact.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None, None

    inline_comment_match = re.search(r"\s+#", line)
    if inline_comment_match:
        package_part = line[: inline_comment_match.start()].strip()
        comment_start = line.find("#", inline_comment_match.start())
        comment = line[comment_start + 1 :].strip() if comment_start >= 0 else None
    else:
        package_part = stripped
        comment = None

    return (package_part if package_part else None), comment


def _parse_legacy_packages_file(file_path: Path) -> list[tuple[str, str | None, str | None]]:
    result: list[tuple[str, str | None, str | None]] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        package_spec, comment = _parse_legacy_line(raw_line)
        if not package_spec:
            continue
        if "|" in package_spec:
            name, version = package_spec.split("|", maxsplit=1)
            version = version.strip() or None
        else:
            name, version = package_spec, None
        result.append((name.strip(), version, comment))
    return result


def _parse_legacy_channels_file(file_path: Path) -> set[str]:
    channels: set[str] = set()
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        channel, _ = _parse_legacy_line(raw_line)
        if channel:
            channels.add(channel)
    return channels


def _resolve_build_step_name(build_step_dir_name: str) -> str:
    return BUILD_STEP_ALIASES.get(build_step_dir_name, build_step_dir_name)


def _discover_build_steps(flavor_path: Path) -> dict[str, Path]:
    build_step_dirs: dict[str, Path] = {}

    flavor_base_path = flavor_path / "flavor_base"
    if flavor_base_path.is_dir():
        for child in sorted(flavor_base_path.iterdir()):
            if child.is_dir() and (child / "packages").is_dir():
                build_step_dirs[child.name] = child

    root_customization = flavor_path / "flavor_customization"
    if root_customization.is_dir() and (root_customization / "packages").is_dir():
        build_step_dirs["flavor_customization"] = root_customization

    return build_step_dirs


def _convert_build_step(build_step_name: str, build_step_path: Path) -> BuildStep | None:
    packages_dir = build_step_path / "packages"
    files = sorted([p for p in packages_dir.iterdir() if p.is_file()])
    if not files:
        return None

    apt_packages: list[AptPackage] = []
    pip_packages: list[PipPackage] = []
    r_packages: list[RPackage] = []
    conda_packages: list[CondaPackage] = []
    conda_channels: set[str] = set()
    has_package_without_version = False

    for package_file in files:
        filename = package_file.name
        parsed = _parse_legacy_packages_file(package_file)

        if filename.startswith("apt_get_packages"):
            for name, version, comment in parsed:
                has_package_without_version = has_package_without_version or (version is None)
                apt_packages.append(AptPackage(name=name, version=version, comment=comment))
            continue

        if filename in {"python3_pip_packages", "python2_pip_packages"}:
            for name, version, comment in parsed:
                has_package_without_version = has_package_without_version or (version is None)
                pip_packages.append(PipPackage(name=name, version=version, comment=comment))
            continue

        if filename in {"cran_packages", "r_cran_packages"}:
            for name, version, comment in parsed:
                has_package_without_version = has_package_without_version or (version is None)
                r_packages.append(RPackage(name=name, version=version, comment=comment))
            continue

        if filename in {"conda_packages", "conda_package"}:
            for name, version, comment in parsed:
                has_package_without_version = has_package_without_version or (version is None)
                conda_packages.append(
                    CondaPackage(name=name, version=version, comment=comment)
                )
            continue

        if filename == "conda_channels":
            conda_channels.update(_parse_legacy_channels_file(package_file))
            continue

        raise ValueError(
            f"Unsupported legacy package file '{package_file}'. "
            "Please extend the migration mapping for this file type."
        )

    phases: list[Phase] = []
    if apt_packages:
        phases.append(
            Phase(
                name=PHASE_NAME_BY_INSTALLER["apt"],
                apt=AptPackages(packages=apt_packages),
            )
        )
    if pip_packages:
        phases.append(
            Phase(
                name=PHASE_NAME_BY_INSTALLER["pip"],
                pip=PipPackages(packages=pip_packages),
            )
        )
    if r_packages:
        phases.append(
            Phase(
                name=PHASE_NAME_BY_INSTALLER["r"],
                r=RPackages(packages=r_packages),
            )
        )
    if conda_packages or conda_channels:
        conda_binary = _detect_conda_binary(build_step_path)
        phases.append(
            Phase(
                name=PHASE_NAME_BY_INSTALLER["conda"],
                conda=CondaPackages(
                    channels=conda_channels or None,
                    packages=conda_packages,
                    binary=conda_binary,
                ),
            )
        )

    if not phases:
        return None

    return BuildStep(
        name=build_step_name,
        phases=phases,
        validation_cfg=ValidationConfig(version_mandatory=not has_package_without_version),
    )


def _normalize_conda_binary_value(conda_binary_value: str) -> str:
    return conda_binary_value.strip().strip("\"'").rstrip(",")


def _detect_conda_binary(build_step_path: Path) -> CondaBinary:
    dockerfile_path = build_step_path / "Dockerfile"
    if not dockerfile_path.is_file():
        raise ValueError(
            f"Conda packages found in '{build_step_path}', but Dockerfile is missing. "
            "Cannot determine conda binary."
        )

    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    matches = re.findall(
        r"install_via_conda\.pl.*?--conda-binary\s+(\"[^\"]+\"|'[^']+'|\S+)",
        dockerfile,
        flags=re.DOTALL,
    )
    if not matches:
        raise ValueError(
            f"Conda packages found in '{build_step_path}', but no '--conda-binary' "
            "argument was found in Dockerfile."
        )

    normalized = {_normalize_conda_binary_value(value) for value in matches}
    if len(normalized) > 1:
        raise ValueError(
            f"Multiple '--conda-binary' values found in '{dockerfile_path}': "
            f"{sorted(normalized)}"
        )

    conda_binary_value = next(iter(normalized))
    if conda_binary_value not in CONDA_BINARY_MAPPING:
        raise ValueError(
            f"Unsupported --conda-binary value '{conda_binary_value}' in '{dockerfile_path}'. "
            f"Supported values: {sorted(CONDA_BINARY_MAPPING.keys())}"
        )
    return CONDA_BINARY_MAPPING[conda_binary_value]


def _sort_build_steps(build_steps: list[BuildStep]) -> list[BuildStep]:
    mapping_order = {name: idx for idx, name in enumerate(BUILD_STEP_MAPPING.keys())}
    return sorted(build_steps, key=lambda bs: (mapping_order.get(bs.name, 999), bs.name))


def _read_micromamba_version_from_conda_deps(flavor_path: Path) -> str | None:
    dockerfile_path = flavor_path / "flavor_base" / "conda_deps" / "Dockerfile"
    if not dockerfile_path.is_file():
        return None
    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    match = re.search(r"^\s*ENV\s+MICROMAMBA_VERSION\s*=\s*([^\s#]+)\s*$", dockerfile, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip().strip("\"'")


def _create_conda_tools_build_step(flavor_path: Path) -> BuildStep | None:
    micromamba_version = _read_micromamba_version_from_conda_deps(flavor_path)
    if not micromamba_version:
        return None
    return BuildStep(
        name="conda",
        phases=[
            Phase(
                name="install_micromamba",
                tools=Tools(
                    micromamba=Micromamba(version=micromamba_version, root_prefix=Path("/opt/conda"))
                ),
            )
        ],
    )


def migrate_flavor(flavor_path: Path) -> None:
    output_build_steps: dict[Path, list[BuildStep]] = defaultdict(list)
    grouped_build_steps: dict[str, list[BuildStep]] = defaultdict(list)

    for raw_build_step_name, build_step_path in _discover_build_steps(flavor_path).items():
        normalized_name = _resolve_build_step_name(raw_build_step_name)
        converted = _convert_build_step(normalized_name, build_step_path)
        if converted:
            grouped_build_steps[normalized_name].append(converted)

    for build_step_name, build_steps in grouped_build_steps.items():
        if build_step_name not in BUILD_STEP_MAPPING:
            raise ValueError(
                f"Build step '{build_step_name}' has package dependencies but is not mapped in BUILD_STEP_MAPPING."
            )
        all_phases: list[Phase] = []
        for build_step in build_steps:
            all_phases.extend(build_step.phases)
        version_mandatory = not any(
            not build_step.validation_cfg.version_mandatory for build_step in build_steps
        )
        output_build_steps[BUILD_STEP_MAPPING[build_step_name]].append(
            BuildStep(
                name=build_step_name,
                phases=all_phases,
                validation_cfg=ValidationConfig(version_mandatory=version_mandatory),
            )
        )

    conda_tools_build_step = _create_conda_tools_build_step(flavor_path)
    if conda_tools_build_step:
        output_build_steps[PUBLIC_PACKAGE_FILE].append(conda_tools_build_step)

    for package_file in {PUBLIC_PACKAGE_FILE, INTERNAL_PACKAGE_FILE}:
        package_file_path = flavor_path / package_file
        package_file_path.parent.mkdir(parents=True, exist_ok=True)
        build_steps = _sort_build_steps(output_build_steps.get(package_file, []))
        if not build_steps:
            if package_file_path.exists():
                package_file_path.unlink()
            continue
        package_file_model = PackageFile(
            build_steps=build_steps
        )
        package_file_path.write_text(
            to_yaml_str(package_file_model),
            encoding="utf-8",
        )


def migrate_flavors(flavors_dir: Path) -> None:
    for flavor_path in sorted([p for p in flavors_dir.iterdir() if p.is_dir()]):
        migrate_flavor(flavor_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate all flavors from legacy package files to packages.yml format."
    )
    parser.add_argument(
        "--flavors-dir",
        type=Path,
        default=FLAVORS_DIR,
        help="Path to flavors root directory (default: ./flavors).",
    )
    args = parser.parse_args()
    migrate_flavors(args.flavors_dir)


if __name__ == "__main__":
    main()
