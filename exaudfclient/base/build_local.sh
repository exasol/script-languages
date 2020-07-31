source .env
export EXAUDF_BASEPATH="$PWD/bazel-bin"
bash build.sh $*
