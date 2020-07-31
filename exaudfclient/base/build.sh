if [ -z "$EXAUDF_BASEPATH" ]
then
      export EXAUDF_BASEPATH=/exaudf
fi
bazel build $*
