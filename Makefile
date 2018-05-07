## This makefile is inspired by https://github.com/jupyter/docker-stacks/blob/master/Makefile
## Builds are tracked via build-stamps in $BUILD_DIR.
## Dependencies for each image are created automatically for all files tracked by git.

BUILD_DIR := build

DOCKER_LOG := $(BUILD_DIR)/docker.log

REGISTRY_VERSION := latest
GIT_VERSION := $(shell git rev-parse --short --verify HEAD)

## List images manually. Keep in sync with dependency tree below.
ALL_IMAGES := linux_container exaudfclient

## Images that should be published in the registry.
OUTPUT_IMAGES:= exaudfclient

help:
	@echo
	@echo '  all                    - build and export all output images'
	@echo '  build-all              - build all images'
	@echo '  export                 - export all output images'
	@echo '  export/<image dirname> - export image'
	@echo '  build/<image dirname>  - builds the latest image and all prerequisite images'
	@echo '  clean                  - remove all build stamps'
	@echo
	@echo '  output images: $(OUTPUT_IMAGES)'
	@echo '  output of failed docker runs can be found in $(DOCKER_LOG).IMAGE_NAME'

build/%: DARGS?=

build-all: $(patsubst %, $(BUILD_DIR)/%, $(ALL_IMAGES))

all: build-all export-all

clean:
	rm -rf $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

## Build rule for all docker stacks
$(BUILD_DIR)/%: | $(BUILD_DIR)
	docker build $(DARGS) --no-cache=true --rm --force-rm -t $(notdir $@):latest ./$(notdir $@) > $(DOCKER_LOG).$(notdir $@) 2>&1
	@touch $(BUILD_DIR)/$(notdir $@)
	@rm $(DOCKER_LOG).$(notdir $@)

## Dependency tree of containers
$(BUILD_DIR)/exaudfclient: $(BUILD_DIR)/linux_container

## Create dependencies for each container: Depend on all files tracked by git.
define generateDependencies
$(BUILD_DIR)/$(1): $(shell git ls-files $(1))
endef

export: $(patsubst %, push/%, $(OUTPUT_IMAGES))

export/%:
	docker run --name $(notdir $@)_container $(notdir $@) > $(DOCKER_LOG).$(notdir $@)_container 2>&1
	docker stop $(notdir $@)_container >> $(DOCKER_LOG).$(notdir $@)_container 2>&1
	docker export $(notdir $@)_container -o exports/$(notdir $@).tar 2>> $(DOCKER_LOG).$(notdir $@)_container
	docker rm $(notdir $@)_container >> $(DOCKER_LOG).$(notdir $@)_container 2>&1
	mkdir exports/$(notdir $@)_tmp >> $(DOCKER_LOG).$(notdir $@)_container 2>&1
	tar xf exports/$(notdir $@).tar -C exports/$(notdir $@)_tmp --exclude=dev --exclude=proc >> $(DOCKER_LOG).$(notdir $@)_container 2>&1
	mkdir exports/$(notdir $@)_tmp/conf exports/$(notdir $@)_tmp/proc exports/$(notdir $@)_tmp/dev >> $(DOCKER_LOG).$(notdir $@)_container 2>&1
	rm exports/$(notdir $@)_tmp/etc/resolv.conf exports/$(notdir $@)_tmp/etc/hosts >> $(DOCKER_LOG).$(notdir $@)_container 2>&1
	ln -s /conf/resolv.conf exports/$(notdir $@)_tmp/etc/resolv.conf >> $(DOCKER_LOG).$(notdir $@)_container 2>&1
	ln -s /conf/hosts exports/$(notdir $@)_tmp/etc/hosts >> $(DOCKER_LOG).$(notdir $@)_container 2>&1
	bash -c 'cd exports/$(notdir $@)_tmp && tar --numeric-owner --owner=0 --group=0 -zcf ../$(notdir $@).tar.gz *' >> $(DOCKER_LOG).$(notdir $@)_container 2>&1
	rm exports/$(notdir $@).tar >> $(DOCKER_LOG).$(notdir $@)_container 2>&1
	rm -rf exports/$(notdir $@)_tmp >> $(DOCKER_LOG).$(notdir $@)_container 2>&1
	@rm $(DOCKER_LOG).$(notdir $@)_container >> $(DOCKER_LOG).$(notdir $@)_container 2>&1

$(foreach container,$(ALL_IMAGES),$(eval $(call generateDependencies,$(container))))

.PHONY: all build-all help clean export-all
