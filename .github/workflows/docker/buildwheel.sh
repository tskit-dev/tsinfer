#!/bin/bash
DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DOCKER_DIR/shared.env"

set -e -x

ARCH=`uname -p`
echo "arch=$ARCH"

# We're running as root in the docker container so git commands issued by
# setuptools_scm will fail without this:
git config --global --add safe.directory /project
# Fetch the full history as we'll be missing tags otherwise.
git fetch --unshallow
for V in "${PYTHON_VERSIONS[@]}"; do
    git reset --hard
    git clean -fd
    PYBIN=/opt/python/$V/bin
    rm -rf build/       # Avoid lib build by narrow Python is used by wide python
    # Instead of letting setup.py install a newer numpy we install it here
    # using the oldest supported version for ABI compatibility
    $PYBIN/python -m venv env
    source env/bin/activate
    $PYBIN/python -m pip install --upgrade build
    SETUPTOOLS_SCM_DEBUG=1 $PYBIN/python -m build
done

cd dist
for whl in *.whl; do
    auditwheel repair "$whl"
    rm "$whl"
done