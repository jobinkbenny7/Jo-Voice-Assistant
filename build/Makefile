# It's necessary to link sh -> bash.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

#ignore make's built-in rules.
MAKEFLAGS += --no-builtin-rules

# Constants used throughout.
.EXPORT_ALL_VARIABLES:
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))
File := $(or $(file), $(CURRENT_DIR)/ipredict)
OUT_DIR ?= $(CURRENT_DIR)/_output
BIN_DIR := $(OUT_DIR)/bin
BUF_VERSION := v1.34.0
PROTOC_GEN_GO_VERSION := v1.34.2
PROTOC_GEN_GO_GRPC_VERSION := v1.4.0
PATH := $(BIN_DIR):${PATH}
BUF_URL :=  https://github.com/bufbuild/buf/releases/download/$(BUF_VERSION)/buf-Linux-x86_64
PROTOC_GEN_GO_URL := https://github.com/protocolbuffers/protobuf-go/releases/download/$(PROTOC_GEN_GO_VERSION)/protoc-gen-go.$(PROTOC_GEN_GO_VERSION).linux.amd64.tar.gz
PROTOC_GEN_GO_GRPC_URL := https://github.com/grpc/grpc-go/releases/download/cmd/protoc-gen-go-grpc/$(PROTOC_GEN_GO_GRPC_VERSION)/protoc-gen-go-grpc.$(PROTOC_GEN_GO_GRPC_VERSION).linux.amd64.tar.gz
GORELEASER_URL :=  https://github.com/goreleaser/goreleaser/releases/download/v1.26.2/goreleaser-1.26.2-1-x86_64.pkg.tar.zst
Protofile := ipredict/gen_code
Protofile_TEST := ipredict-test.proto
TEST_PATH := QA

gen-tools/bin: $(BIN_DIR)/buf $(BIN_DIR)/protoc-gen-go $(BIN_DIR)/protoc-gen-go-grpc

$(BIN_DIR)/buf:
	mkdir -p "$(BIN_DIR)" && \
    curl -L $(BUF_URL) -o "$(BIN_DIR)/buf" && \
	chmod u+x "$(BIN_DIR)/buf" && \
	touch "$(BIN_DIR)/buf" > /dev/null 2>&1

$(BIN_DIR)/protoc-gen-go:
	mkdir -p "$(BIN_DIR)" && \
    curl -L $(PROTOC_GEN_GO_URL) -o "$@".tar && \
	tar xzf "$@".tar --directory "$(BIN_DIR)" && \
	chmod u+x "$@" && \
	stat "$@" > /dev/null 2>&1

$(BIN_DIR)/protoc-gen-go-grpc:
	mkdir -p "$(BIN_DIR)" && \
    curl -L $(PROTOC_GEN_GO_GRPC_URL) -o "$@".tar && \
	tar -xzf "$@".tar --directory "$(BIN_DIR)" && \
	chmod u+x "$@" && \
	stat "$@" > /dev/null 2>&1

.PHONY: buf/update
buf/update: gen-tools/bin
	@buf mod update

.PHONY: buf/generate
buf/generate: gen-tools/bin
	@buf generate --path $(Protofile)

.PHONY: buf/generate_ipredict_testing
buf/generate_ipredict_testing: gen-tools/bin
	@buf generate --path $(Protofile_TEST)  --output $(TEST_PATH)

#download goreleaser
$(BIN_DIR)/goreleaser:
	mkdir -p $(BIN_DIR) && \
    mkdir -p $(BIN_DIR)/tmp-goreleaser && \
    curl -L $(GORELEASER_URL) -o $(BIN_DIR)/tmp-goreleaser/goreleaser.tar && \
    tar -xf $(BIN_DIR)/tmp-goreleaser/goreleaser.tar -C $(BIN_DIR)/tmp-goreleaser && \
	chmod u+x $(BIN_DIR)/tmp-goreleaser/usr/bin/goreleaser && \
    mv $(BIN_DIR)/tmp-goreleaser/usr/bin/goreleaser $(BIN_DIR) && \
    rm -rf $(BIN_DIR)/tmp-goreleaser && \
	touch $@

#install poetry
$(BIN_DIR)/poetry:
	mkdir -p $(BIN_DIR) && \
    export POETRY_HOME=$(OUT_DIR) && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    touch $@

#release locally
.PHONY: release-local
release-local: $(BIN_DIR)/poetry $(BIN_DIR)/goreleaser
	@goreleaser release --clean --snapshot

#build
.PHONY: build
build: $(BIN_DIR)/poetry
	@poetry build -o $(OUT_DIR)

#build
.PHONY: dev-run
dev-run: $(BIN_DIR)/poetry
	@poetry run python $(ARGS)

#release
.PHONY: release
release: $(BIN_DIR)/goreleaser
	@goreleaser release --clean


.PHONY: install_lint_tools pip_audit black isort pylint codespell pyright bandit radon radon-ci verify-fix

PYTHONPATH := $(CURRENT_DIR)
EXCLUDE_DEFAULT_DIRS := gen_code
EXCLUDE_DIRS := $(EXCLUDE_DEFAULT_DIRS) $(if $(exclude),$(exclude))
EXCLUDE_RULES := -type f -name "*.py" $(foreach dir,$(EXCLUDE_DIRS),-not -path "*/$(dir)/*")

install_lint_tools: $(BIN_DIR)/poetry
	@echo "-----------------------------------------------------------------"
	@echo "installing linting tools"
	@echo "-----------------------------------------------------------------"
	@poetry install --no-root --with lint

pip_audit:
	@echo "-----------------------------------------------------------------"
	@echo "Running library checks"
	@echo "-----------------------------------------------------------------"
	@poetry run pip-audit

black:
	@echo "-----------------------------------------------------------------"
	@echo "Running Black"
	@echo "-----------------------------------------------------------------"
	@find $(File) $(EXCLUDE_RULES) -exec poetry run black {} +

isort:
	@echo "-----------------------------------------------------------------"
	@echo "Running iSort"
	@echo "-----------------------------------------------------------------"
	@find $(File) $(EXCLUDE_RULES) -exec poetry run isort {} +

pylint:
	@echo "-----------------------------------------------------------------"
	@echo "Running Pylint"
	@echo "-----------------------------------------------------------------"
	@echo "$(File)"
	@find $(File) $(EXCLUDE_RULES) -exec poetry run pylint --disable=W0718,W0702,W0719,W0707,E0401,E0611 {} +

codespell:
	@echo "-----------------------------------------------------------------"
	@echo "Running codespell"
	@echo "-----------------------------------------------------------------"
	@find $(File) $(EXCLUDE_RULES) -exec poetry run codespell {} +

pyright:
	@echo "-----------------------------------------------------------------"
	@echo "Running Pyright"
	@echo "-----------------------------------------------------------------"
	@find $(File) $(EXCLUDE_RULES) -exec poetry run pyright --level=warning {} +

bandit:
	@echo "-----------------------------------------------------------------"
	@echo "Running bandit"
	@echo "-----------------------------------------------------------------"
	@find $(File) $(EXCLUDE_RULES) -exec poetry run bandit -r {} +

radon:
	@echo "-----------------------------------------------------------------"
	@echo "Running radon"
	@echo "-----------------------------------------------------------------"
	@echo "Cyclomatic Complexity"
	@find $(File) $(EXCLUDE_RULES) -exec poetry run radon cc -n C {} +
	@echo "-----------------------------------------------------------------"
	@echo "Managebility"
	@find $(File) $(EXCLUDE_RULES) -exec poetry run radon mi -n C {} +
	@echo "-----------------------------------------------------------------"

radon-ci:
	@echo "-----------------------------------------------------------------"
	@echo "Running radon"
	@echo "-----------------------------------------------------------------"
	@echo "Cyclomatic Complexity"
	@c=$$(find $(File) $(EXCLUDE_RULES) -exec poetry run radon cc -n C {} \; | wc -l); \
	if [ "$$c" -ne 0 ]; then \
		echo "Exiting with an error: Issue with cyclomatic complexity. Please check 'make radon'"; \
		exit 1; \
	fi
	@echo "-----------------------------------------------------------------"
	@echo "Manageability"
	@c=$$(find $(File) $(EXCLUDE_RULES) -exec radon mi -n C {} \; | wc -l); \
	if [ "$$c" -ne 0 ]; then \
		echo "Exiting with an error: Issue with Manageability. Please check 'make radon'"; \
		exit 1; \
	fi

verify: install_lint_tools pylint
	@echo "-----------------------------------------------------------------"
	@echo "Verify finished"
	@echo "Run 'make verify-fix' to fix format the files"
	@echo "-----------------------------------------------------------------"

verify-fix: black isort
	@echo "-----------------------------------------------------------------"
	@echo "Finished running Formatting tools"
	@echo "-----------------------------------------------------------------"

help:
	@echo "-----------------------------------------------------------------"
	@echo "Makefile Help"
	@echo "-----------------------------------------------------------------"
	@echo "Available options:"
	@echo "  verify         - Run all tools"
	@echo "  verify-fix     - Run Formatting tools"
	@echo "  install_lint_tools  - Install tools and needed dependencies"
	@echo "  pip_audit      - Checks for library vulnerabilities"
	@echo "  black          - Formats the style of the code"
	@echo "  isort          - Sorts import statements"
	@echo "  pylint         - Runs Pylint for Finding syntax errors"
	@echo "  codespell      - Checks for Typos/Spelling errors"
	@echo "  pyright        - Type Checks the code"
	@echo "  bandit         - Checks for any vulnerability points in code"
	@echo "  radon          - Checks the complexity of individual functions below C"
	@echo "  help           - Display this help menu"
	@echo "  file           - Include the path of the file or the folder"
	@echo "  exclude        - Exclude certain directories (for multiple directories use double quotes)"
	@echo "-----------------------------------------------------------------"
