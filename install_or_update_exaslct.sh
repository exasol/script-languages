#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "$*" 1>&2
  exit 1
}
download_raw_file_from_github() {
  local repo=$1
  local ref=$2
  local remote_file_path=$3
  local local_file_path=$4
  local url="https://api.github.com/repos/$repo/contents/$remote_file_path?ref=$ref"
  local arguments=(-s -H 'Accept: application/vnd.github.v3.raw' -L "$url" -o "$local_file_path")
  if [ -z "${GITHUB_TOKEN-}" ]; then
    curl "${arguments[@]}"
  else
    curl -H "Authorization: token $GITHUB_TOKEN" "${arguments[@]}"
  fi
}
download_and_verify_raw_file_from_github() {
  local repo=$1
  local ref=$2
  local remote_file_path=$3
  local file_name=${remote_file_path##*/}
  local dir_path=${remote_file_path%$file_name}
  local checksum_file_name="${file_name}.sha512sum"
  local remote_checksum_file_path="${dir_path}checksums/$checksum_file_name"

  download_raw_file_from_github "$repo" "$ref" "$remote_file_path" "$file_name" ||
    die "ERROR: Could not download '$remote_file_path' from the github repository '$repo' at ref '$ref'."
  download_raw_file_from_github "$repo" "$ref" "$remote_checksum_file_path" "$checksum_file_name" ||
    die "ERROR: Could not download the checksum for '$remote_file_path' from the github repository '$repo' at ref '$ref'."
  sha512sum --check "${checksum_file_name}" ||
    die "ERROR: Could not verify the checksum for '$remote_file_path' from the github repository '$repo' at ref '$ref'."

}

main() {
  local exaslct_git_ref="latest"
  if [ -n "${1-}" ]; then
    exaslct_git_ref="$1"
  fi

  local repo="exasol/script-languages-container-tool"
  tmp_directory_for_installer="$(mktemp -d)"
  trap 'rm -rf -- "$tmp_directory_for_installer"' EXIT

  local installer_file_name="exaslct_installer.sh"

  pushd "$tmp_directory_for_installer" &>/dev/null

  download_and_verify_raw_file_from_github "$repo" "$exaslct_git_ref" "installer/$installer_file_name"

  popd &>/dev/null

  bash "$tmp_directory_for_installer/$installer_file_name" "$exaslct_git_ref"

}

main "${@}"
