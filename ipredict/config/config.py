# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring,line-too-long,global-statement

import logging
import os
import sys

import requests
from dotenv import load_dotenv
from requests.exceptions import ConnectTimeout

logger = logging.getLogger(__name__)


IPREDICT_IS_LOCAL_TEST = False
IPREDICT_DATABASE_TYPE = "QuestDB"
IPREDICT_ANOMALY_START_END_DATE_MANDATORY = True
IPREDICT_AGGREGATED_FORECAST_DATA = True
IPREDICT_FORECAST_REPLACE_EXPECTED_WITH_ACTUAL_IF_EXIST = True
IPREDICT_MAX_BUDGET_YEAR_COUNT = 15
IPREDICT_MIN_BUDGET_YEAR_COUNT = 2
IPREDICT_FORECAST_THRESHOLD = 0.1
IPREDICT_QUESTDB_HOST = "127.0.0.0.1"
IPREDICT_QUESTDB_HOSTPORT = "9000"
IPREDICT_SQL_PASSWD = None
IPREDICT_SQL_HOST = None
IPREDICT_SQL_PORT = None
IPREDICT_SQL_HOSTNAME = None
STORAGE_PATH = "./result_folder"
CONNECTION_FAIL_CODE = 123
VALUE_ERROR_CODE = 147
PLOT_POINTS = 100

QUESTDB_CONNECTION_STRING = ""
QUESTDB_CONNECTION_URL = ""


def init_config(env_path):
    global IPREDICT_DATABASE_TYPE
    global IPREDICT_QUESTDB_HOST, IPREDICT_QUESTDB_HOSTPORT
    global QUESTDB_CONNECTION_STRING, QUESTDB_CONNECTION_URL
    global IPREDICT_SQL_PASSWD, IPREDICT_SQL_HOST, IPREDICT_SQL_PORT, IPREDICT_SQL_HOSTNAME
    global STORAGE_PATH
    load_dotenv(dotenv_path=env_path)

    logger.info("ipredict_is_local_test-->%s", os.environ.get("IPREDICT_IS_LOCAL_TEST"))

    IPREDICT_DATABASE_TYPE = os.environ.get("IPREDICT_DATABASE_TYPE", "QuestDB")
    logger.info("database_type %s", IPREDICT_DATABASE_TYPE)
    IPREDICT_QUESTDB_HOST = os.environ.get("IPREDICT_QUESTDB_HOST")
    logger.info("quest db host ip -  %s", IPREDICT_QUESTDB_HOST)
    IPREDICT_QUESTDB_HOSTPORT = os.environ.get("IPREDICT_QUESTDB_HOSTPORT")
    QUESTDB_CONNECTION_STRING = (
        f"http::addr={IPREDICT_QUESTDB_HOST}:{IPREDICT_QUESTDB_HOSTPORT};"
    )
    QUESTDB_CONNECTION_URL = (
        f"http://{IPREDICT_QUESTDB_HOST}:{IPREDICT_QUESTDB_HOSTPORT}/exec"
    )
    logger.info("QuestDB connection URL: %s", QUESTDB_CONNECTION_URL)

    IPREDICT_SQL_PASSWD = os.environ.get("IPREDICT_SQL_PASSWD")
    IPREDICT_SQL_HOST = os.environ.get("IPREDICT_SQL_HOST")
    IPREDICT_SQL_PORT = os.environ.get("IPREDICT_SQL_PORT")
    IPREDICT_SQL_HOSTNAME = os.environ.get("IPREDICT_SQL_HOSTNAME")
    STORAGE_PATH = os.environ.get("STORAGE_PATH")
    query = "show tables;"
    params = {"query": query}
    try:
        response = requests.get(QUESTDB_CONNECTION_URL, params=params, timeout=60)

        if response.status_code != 200:
            logger.error(
                "QuestDB query failed. Status code: %s, Response: %s",
                response.status_code,
                response.text,
            )
            raise ValueError(
                f"QuestDB query failed with status code {response.status_code}: {response.text}"
            )

    except ConnectTimeout:
        logger.exception("Database connection failed for %s", QUESTDB_CONNECTION_URL)
        sys.exit(CONNECTION_FAIL_CODE)
