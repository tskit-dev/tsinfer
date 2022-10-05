
# Need to set PYTHONPATH so that we pick up the local tsinfer
PYPATH=$(shell pwd)/../
TSINF_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import tsinfer; print(tsinfer.__version__.split("+")[0])')

BUILDDIR      = _build

all: dev

dev:
	PYTHONPATH=${PYPATH} ./build.sh

dist:
	@echo Building distribution for tskit version ${TSINF_VERSION}
	sed -i -e s/__TSINFER_VERSION__/${TSINF_VERSION}/g _config.yml
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR $(BUILDDIR)
