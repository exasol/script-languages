if [ -z "$EXAUDF_BASEPATH" ]
then
      echo EXAUDF_BASEPATH=/exaudf
fi
bazel build $*
