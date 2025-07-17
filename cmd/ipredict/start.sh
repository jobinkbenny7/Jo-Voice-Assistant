#!/usr/bin/env bash

set -e

case "$1" in
  api)
    shift;
    # Run API Server
    exec python3 -m ipredict.api.server serve $@
    # exec python3 QA/ipredict/server_test.py 
    ;;

  jobs)
    shift;
    # Run data process as job
    exec python3 -m ipredict.data_processor.processor $@
    ;;

  forecast)
    shift;
    # Run data process as async API
    exec python3 -m ipredict.data_processor.processor forecast $@
    ;;

  *)
    echo "Usage: $0 {api/jobs/forecast}" >&2
    exit 1
    ;;
esac
