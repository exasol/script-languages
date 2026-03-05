from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_FLAVORS_DIR = Path(__file__).resolve().parents[1] / "flavors"

INSTALL_LINE_PATTERN = re.compile(
    r"^\s*RUN\s+/scripts/install_scripts/install_via_(apt|pip|r_remotes|conda)\.pl\b.*$"
)
FROM_UBUNTU_PATTERN = re.compile(r"^(\s*FROM\s+)ubuntu:(22\.04|24\.04)(\b.*)$")


def _discover_build_step_dirs(flavor_path: Path) -> list[Path]:
    result: list[Path] = []

    flavor_base = flavor_path / "flavor_base"
    if flavor_base.is_dir():
        for child in sorted(flavor_base.iterdir()):
            if (child / "Dockerfile").is_file():
                result.append(child)

    flavor_customization = flavor_path / "flavor_customization"
    if (flavor_customization / "Dockerfile").is_file():
        result.append(flavor_customization)

    return result


def _rewrite_dockerfile(dockerfile_path: Path, build_step_name: str) -> bool:
    changed = False
    rewritten_lines: list[str] = []
    copied_package_file = False
    inserted_exaslpm_install = False
    package_file_in_image = f"/build_info/packages/{build_step_name}_packages.yml"
    copy_line = f"COPY {build_step_name}_packages.yml {package_file_in_image}"
    exaslpm_line = (
        f"RUN exaslpm install --package-file {package_file_in_image} "
        f"--build-step {build_step_name}"
    )
    legacy_copy_line = f"COPY {build_step_name}_packages.yml /build_info/{build_step_name}_packages.yml"
    legacy_exaslpm_line = (
        f"RUN exaslpm install --package-file /build_info/{build_step_name}_packages.yml "
        f"--build-step {build_step_name}"
    )
    old_packages_copy_pattern = re.compile(
        rf"^\s*COPY\s+{re.escape(build_step_name)}/packages(?:/|\b).*\s+/build_info/packages(?:/|\b).*$"
    )
    old_conda_activate_copy_line = (
        "COPY scripts/virtual_environment/_activate_current_conda_env.sh "
        "/usr/local/bin/_activate_current_env.sh"
    )
    remove_conda_deps_line_patterns = [
        re.compile(r'^\s*ENV\s+ENV_NAME="base"\s*$'),
        re.compile(r'^\s*ENV\s+MAMBA_ROOT_PREFIX="/opt/conda"\s*$'),
        re.compile(r'^\s*ENV\s+MAMBA_EXE="/bin/micromamba"\s*$'),
        re.compile(r"^\s*ENV\s+MAMBA_DOCKERFILE_ACTIVATE=1\s*$"),
        re.compile(r'^\s*SHELL\s+\["/bin/bash",\s*"-l",\s*"-c"\]\s*$'),
        re.compile(r"^\s*ENV\s+MICROMAMBA_VERSION=.*$"),
        re.compile(r'^\s*RUN\s+/scripts/install_scripts/install_micromamba\.sh\s+"\$MICROMAMBA_VERSION"\s*$'),
    ]
    remove_build_deps_bazel_env_patterns = [
        re.compile(r'^\s*ENV\s+BAZEL_PACKAGE_VERSION=.*$'),
        re.compile(r'^\s*ENV\s+BAZEL_PACKAGE_FILE=.*$'),
        re.compile(r'^\s*ENV\s+BAZEL_PACKAGE_URL=.*$'),
    ]

    for line in dockerfile_path.read_text(encoding="utf-8").splitlines():
        from_match = FROM_UBUNTU_PATTERN.match(line)
        if from_match:
            line = (
                f"{from_match.group(1)}exasol/slc_base:{from_match.group(2)}"
                f"{from_match.group(3)}"
            )
            changed = True

        if old_packages_copy_pattern.match(line):
            changed = True
            continue

        if line.strip() == old_conda_activate_copy_line:
            changed = True
            continue

        if build_step_name == "conda_deps" and any(
            pattern.match(line) for pattern in remove_conda_deps_line_patterns
        ):
            changed = True
            continue

        if build_step_name == "build_deps" and any(
            pattern.match(line) for pattern in remove_build_deps_bazel_env_patterns
        ):
            changed = True
            continue

        if INSTALL_LINE_PATTERN.match(line):
            if not copied_package_file:
                rewritten_lines.append(copy_line)
                copied_package_file = True
            if not inserted_exaslpm_install:
                rewritten_lines.append(exaslpm_line)
                inserted_exaslpm_install = True
            changed = True
            continue

        if line.strip() == legacy_copy_line:
            line = copy_line
            changed = True

        if line.strip() == legacy_exaslpm_line:
            line = exaslpm_line
            changed = True

        if line.strip() == copy_line:
            if copied_package_file:
                changed = True
                continue
            copied_package_file = True

        if line.strip() == exaslpm_line:
            if inserted_exaslpm_install:
                changed = True
                continue
            inserted_exaslpm_install = True

        rewritten_lines.append(line)

    rewritten_text = "\n".join(rewritten_lines) + "\n"
    if build_step_name == "build_deps":
        bazel_run_block_pattern = re.compile(
            r"""
            ^\s*RUN\ apt-get\ -y\ update\ \&\&\ \\\s*\n
            \s*curl\ -L\ --output\ "\$BAZEL_PACKAGE_FILE"\ "\$BAZEL_PACKAGE_URL"\ \&\&\ \\\s*\n
            \s*apt-get\ install\ -y\ "\./\$BAZEL_PACKAGE_FILE"\ \&\&\ \\\s*\n
            \s*rm\ "\$BAZEL_PACKAGE_FILE"\ \&\&\ \\\s*\n
            \s*apt-get\ -y\ clean\ \&\&\ \\\s*\n
            \s*apt-get\ -y\ autoremove\s*$
            """,
            re.MULTILINE | re.VERBOSE,
        )
        without_bazel_run = re.sub(bazel_run_block_pattern, "", rewritten_text)
        if without_bazel_run != rewritten_text:
            changed = True
            rewritten_text = without_bazel_run

    if changed:
        dockerfile_path.write_text(rewritten_text, encoding="utf-8")
    return changed


def migrate_flavor(flavor_path: Path) -> int:
    changed_files = 0
    for build_step_dir in _discover_build_step_dirs(flavor_path):
        dockerfile = build_step_dir / "Dockerfile"
        if _rewrite_dockerfile(dockerfile, build_step_dir.name):
            changed_files += 1
    return changed_files


def migrate_flavors(flavors_dir: Path) -> int:
    total_changed_files = 0
    for flavor_path in sorted([path for path in flavors_dir.iterdir() if path.is_dir()]):
        total_changed_files += migrate_flavor(flavor_path)
    return total_changed_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate flavor Dockerfiles from install_via_* scripts to exaslpm."
    )
    parser.add_argument(
        "--flavors-dir",
        type=Path,
        default=DEFAULT_FLAVORS_DIR,
        help=f"Path to flavors directory (default: {DEFAULT_FLAVORS_DIR}).",
    )
    args = parser.parse_args()
    changed_files = migrate_flavors(args.flavors_dir)
    print(f"Updated {changed_files} Dockerfile(s).")


if __name__ == "__main__":
    main()
