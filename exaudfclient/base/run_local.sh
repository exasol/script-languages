source .env
export VERBOSE_BUILD="--subcommands --verbose_failures"
bash run.sh --define streaming=true --define python=true --define java=true --define benchmark=true --define r=true $*