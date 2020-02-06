init_pipenv () {
    PIPENV_BIN="$1"
    echo "Using following pipenv executable: $PIPENV_BIN"
    if $PIPENV_BIN --venv &> /dev/null
    then
        echo "Using existing virtual environment"
    else
        echo "Creating new virtual environment and installing dependencies"
        $PIPENV_BIN install
    fi
    if [ ! -f Pipfile.lock ]
    then
        echo "Installing dependencies"
        $PIPENV_BIN install
    fi
}


find_pip_bin () {
    if python3 -m pip list &> /dev/null
    then
        PIP_BIN="python3 -m pip "
    else
        echo "ERROR: cant find pip"
	      exit 1
    fi
}

find_pipenv_bin_via_pip () {
    find_pip_bin
    local PIPENV_LOCATION=$($PIP_BIN show pipenv | grep 'Location:'  | cut -f 2 -d " ")
    local PIPENV_BIN_IN_LOCATION=$($PIP_BIN show -f pipenv | grep 'bin/pipenv$' | awk '{$1=$1};1')
    PIPENV_BIN=$(command -v "$PIPENV_LOCATION/$PIPENV_BIN_IN_LOCATION")
    if [ -z "$PIPENV_BIN" ]
    then
        echo "ERROR: pipenv seems to be installed, but I can't find in the PATH or via pip"
        exit 1
    fi
}


request_install_to_virtual_env () {
    if [ "$PIP_INSTALL" = YES ]
    then
      ANSWER=yes
    else
      echo "Do you want to install pipenv into the current virtual environment: yes/no"
      read ANSWER
    fi
    if [ "$ANSWER" == "yes" ]
    then
        find_pip_bin
        $PIP_BIN install pipenv
        find_pipenv_bin_via_pip
    else
        echo "Aborting"
        exit 1
    fi
}

request_install_to_user () {
    if [ "$PIP_INSTALL" = YES ]
    then
      ANSWER=yes
    else
      echo "Do you want to install pipenv local to the user: yes/no"
      read ANSWER
    fi
    if [ "$ANSWER" == "yes" ]
    then
        find_pip_bin
        $PIP_BIN install --user pipenv
        PIPENV_BIN="$(python3 -m site --user-base)/bin/pipenv"
    else
        echo "Aborting"
        exit 1
    fi
}

request_install () {
    IS_IN_VIRTUAL_ENV=$(python3 -c "import sys; print(hasattr(sys, 'real_prefix'))")
    if [ "$IS_IN_VIRTUAL_ENV" == "True" ]
    then
        request_install_to_virtual_env
    else
        request_install_to_user
    fi
}

discover_pipenv() {
    if [ -z "$PIPENV_BIN" ]
    then
      PIPENV_BIN=$(command -v pipenv)
      if [ -z "$PIPENV_BIN" ]
      then
          find_pip_bin
          if $PIP_BIN show pipenv &> /dev/null
          then
              find_pipenv_bin_via_pip
          else
              request_install
          fi
      fi
    fi
}
