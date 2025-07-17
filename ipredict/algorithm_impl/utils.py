# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring,line-too-long,redefined-builtin,too-many-positional-arguments,too-many-arguments,inconsistent-return-statements

import logging
import os
import pickle
from datetime import datetime

import mysql.connector
import pandas as pd
from dateutil import tz
import requests
from mysql.connector import Error
from questdb.ingress import Sender
from sqlalchemy import create_engine

from ipredict.config import config

FREQ_DICT = {
    "Month": "M",
    "Week": "W",
    "Day": "D",
    "Hour": "H",
    "Minute": "min",
    "Second": "S",
}


def init_logger():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_output = os.environ.get("LOG_OUTPUT", "console").lower()
    log_file_path = os.environ.get("LOG_FILE_PATH", "app.log")

    if log_level == "DEBUG":
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        format = "%(asctime)s - %(levelname)s - %(message)s"

    if log_output == "console":
        logging.basicConfig(level=log_level, format=format)

    elif log_output == "file":
        logging.basicConfig(
            level=log_level, format=format, filename=log_file_path, filemode="w"
        )
    else:
        raise ValueError(
            "Invalid LOG_OUTPUT value. Must be 'console', 'file', or 'server'."
        )


logger = logging.getLogger(__name__)


def check_database_exists(connection, database):
    try:
        cursor = connection.cursor()
        cursor.execute(f"SHOW DATABASES LIKE '{database}'")
        result = cursor.fetchone()
        return result is not None
    except Error as e:
        logger.error("Error: %s", e)
        return False


def check_table_exists(connection, database, table):
    try:
        cursor = connection.cursor()
        if check_database_exists(connection, database) is True:
            cursor.execute(f"USE {database};")
            cursor.execute(f"SHOW TABLES LIKE '{table}';")
            result = cursor.fetchone()
            return result is not None
        return False
    except Error as e:
        logger.error("Error: %s", e)
        return False


def is_table_exists(table):
    if config.IPREDICT_DATABASE_TYPE == "QuestDB":
        return is_questdb_table_exist(table)


def create_database(database_name):
    logger.debug("Calling create_database")
    data_base = mysql.connector.connect(
        host=config.IPREDICT_SQL_HOST,
        user=config.IPREDICT_SQL_HOSTNAME,
        passwd=config.IPREDICT_SQL_PASSWD,
    )
    cursor_object = data_base.cursor()
    cursor_object.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
    data_base.close()


def save_sql_data_df(database, table, data_df, write_mode="replace"):
    logger.debug("function -> save_data_df")
    create_database(database)
    connection_string = f"mysql+pymysql://{config.IPREDICT_SQL_HOSTNAME}:{config.IPREDICT_SQL_PASSWD}@{config.IPREDICT_SQL_HOST}:{config.IPREDICT_SQL_PORT}/"
    engine = create_engine(connection_string)
    db_connection = engine.connect()
    data_df.to_sql(
        name=table, con=engine, schema=database, if_exists=write_mode, index=False
    )
    db_connection.close()


def read_sql_data_df(database, table):
    logger.debug("function -> read_data_df")
    connection_string = f"mysql+pymysql://{config.IPREDICT_SQL_HOSTNAME}:{config.IPREDICT_SQL_PASSWD}@{config.IPREDICT_SQL_HOST}:{config.IPREDICT_SQL_PORT}/"
    engine = create_engine(connection_string)
    db_connection = engine.connect()
    try:
        result_data_frame = pd.read_sql(f"USE {database};", db_connection)
        __ = result_data_frame
    except Exception:
        pass
    query = f"SELECT * FROM {table};"
    sql_df = pd.read_sql_query(query, db_connection)
    db_connection.close()
    return sql_df


def truncate_table(table_name):
    url = f"http://{config.IPREDICT_QUESTDB_HOST}:{config.IPREDICT_QUESTDB_HOSTPORT}/exec?query=truncate+table+'{table_name}'"
    response = requests.get(url, timeout=60)
    if response.status_code == 200:
        logger.info("Table '%s' truncated successfully.", table_name)
    else:
        logger.error("Failed to truncate table '%s'.", table_name)
        logger.debug(response.text)


def is_questdb_table_exist(table_name):
    table_present = False
    query = "show tables;"
    params = {"query": query}
    response = requests.get(config.QUESTDB_CONNECTION_URL, params=params, timeout=60)
    if response.status_code == 200:
        data = response.json()
        for val in data["dataset"]:
            if val[0] == table_name:
                logger.debug("table present")
                table_present = True
    else:
        logger.error("Failed to fetch tables")
    return table_present


def save_questdb_data_df(database, table, data_df, write_mode="replace"):
    table_name = f"{database}_{table}"
    at = "ds"
    if database == "trend_param":
        data_df["ds"] = [datetime.now() for i in range(data_df.shape[0])]
        data_df["ds"] = pd.to_datetime(data_df["ds"])
    if database == "budget":
        at = "DateTime"
    if write_mode == "replace":
        if is_questdb_table_exist(table_name) is True:
            truncate_table(table_name)
    try:
        with Sender.from_conf(config.QUESTDB_CONNECTION_STRING) as sender:
            sender.dataframe(data_df, table_name=table_name, at=at)
    except Exception as e:
        logger.info("Error saving data to QuestDB: %s", e)


def save_data_df(database, table, data_df, write_mode="replace"):
    if config.IPREDICT_DATABASE_TYPE == "QuestDB":
        save_questdb_data_df(database, table, data_df, write_mode)
    else:
        save_sql_data_df(database, table, data_df, write_mode)


def execute_query_get_df(query):
    logger.info(query)
    response = requests.get(
        config.QUESTDB_CONNECTION_URL, params={"query": query}, timeout=60
    )
    logger.debug(response)
    df = pd.DataFrame()
    if response.status_code == 200:
        data = response.json()
        columns = data["columns"]
        dataset = data["dataset"]
        df = pd.DataFrame(dataset, columns=[col["name"] for col in columns])
        try:
            if "timestamp" in df.columns:
                df = df.reset_index(drop=True).rename(columns={"timestamp": "ds"})
            else:
                logger.error("'timestamp' column not found in the DataFrame")
        except Exception as e:
            logger.exception(
                "An unexpected error occurred during reset_index/rename: %s", e
            )
        try:
            if "ds" in df.columns:
                df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
            else:
                logger.error("'ds' column not found in the DataFrame")
        except Exception as e:
            logger.exception(
                "An unexpected error occurred during datetime conversion: %s", e
            )

    else:
        logger.error("Failed to execute query:%s", response.text)
    return df


def get_questdb_data_query_with_start_end_date_mandatory(
    database, table, start_date, end_date
):
    table_name = f"{database}_{table}"
    query = f"SELECT * FROM '{table_name}' WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'"
    return query


def get_questdb_data_query(database, table, start_date, end_date):
    df_start_date = get_first_date_from_questdb_df(
        database=database, table=table, datetime_column="timestamp"
    )
    df_end_date = get_latest_date_from_questdb_df(
        database=database, table=table, datetime_column="timestamp"
    )
    if (start_date is not None) and ((start_date < df_start_date) or (start_date == 0)):
        logger.debug("inside (start_date != None) and (start_date < df_start_date)")
        start_date = df_start_date
    if (end_date is not None) and ((end_date > df_end_date) or (end_date == 0)):
        logger.debug("inside (end_date != None) and (end_date > df_end_date)")
        end_date = df_end_date
    table_name = f"{database}_{table}"
    if (start_date is None) or (end_date is None) or (start_date > end_date):
        query = f"SELECT * FROM '{table_name}'"
        logger.debug(
            """inside (start_date == None) or
              (end_date == None) or
			  (start_date < end_date) -> %s""",
            query,
        )
    else:
        start_date = convert_unix_to_str_date(start_date)
        end_date = convert_unix_to_str_date(end_date)
        query = f"SELECT * FROM '{table_name}' WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'"
        logger.debug("Inside else part %s", query)
    return query


def read_questdb_data_df(database, table, start_date, end_date):
    query = get_questdb_data_query(database, table, start_date, end_date)
    return execute_query_get_df(query)


def read_data_df(database, table, start_date=None, end_date=None):
    if config.IPREDICT_DATABASE_TYPE == "QuestDB":
        return read_questdb_data_df(database, table, start_date, end_date)
    return read_sql_data_df(database, table)


def get_resample_start_end_date_from_input(data_df, time_type, time_interval):
    start_time = list(data_df["ds"])[0]
    end_time = list(data_df["ds"])[-1]
    logger.debug(
        "start_time - %s - type - %s   -----  end_time - %s  - type - %s",
        start_time,
        type(start_time),
        end_time,
        type(end_time),
    )
    input_data_time_diff = get_data_time_diff_in_seconds(data_df["ds"].diff())
    logger.info("time difference - %s", input_data_time_diff)
    second_val = {"W": (7 * 24 * 3600), "D": (24 * 3600), "H": 3600, "min": 60, "S": 1}
    if time_type in second_val:
        output_time_intervel_in_seconds = second_val[time_type] * time_interval
        logger.debug(
            "Value of time_type %s in Seconds -> %s",
            time_type,
            output_time_intervel_in_seconds,
        )
        if output_time_intervel_in_seconds <= input_data_time_diff:
            logger.error(
                "Existing data difference is greater than the requested aggregation"
            )
            return None, None

        dummy_df = pd.DataFrame()
        dummy_df["ds"] = [start_time]
        dummy_df["yhat"] = [2]
        resample_df = resample_data_without_timezone(
            dummy_df,
            time_type,
            "sum",
            time_interval,
            output_col="yhat",
            label="left",
        )
        start_time = list(resample_df["ds"])[0]
        dummy_df["ds"] = [end_time]
        resample_df = resample_data_without_timezone(
            dummy_df,
            time_type,
            "sum",
            time_interval,
            output_col="yhat",
            label="left",
        )
        end_time = list(resample_df["ds"])[0]
        return start_time, end_time
    return start_time, end_time


def get_data_time_diff_in_seconds(dt):
    time_diff = dt.iloc[dt.values.nonzero()[0]].min()
    total_seconds = time_diff.total_seconds()
    return total_seconds


def save_model(model, model_folder_path):
    logger.debug("function -> save_model")
    path = os.environ.get("STORAGE_PATH")
    result_folder_path = f"{path}/{model_folder_path}/"
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    pkl_path = f"{path}/{model_folder_path}/Prophet.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)


def read_model(model_folder_path):
    logger.debug("function -> read_model")
    path = os.environ.get("STORAGE_PATH")
    pkl_path = f"{path}/{model_folder_path}/Prophet.pkl"
    logger.info("pkl_path -> %s", pkl_path)
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        return model


def convert_unix_to_str_date(date_val, time_zone=True):
    dt = datetime.fromtimestamp(date_val)
    if time_zone is True:
        timestamp_str = dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    else:
        timestamp_str = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    return timestamp_str


def convert_str_to_unix_date(date_val, format="%Y-%m-%dT%H:%M:%S.%fZ"):
    date_in_str = int((datetime.strptime(date_val, format)).timestamp())
    return date_in_str


def get_latest_date_from_questdb_df(database, table, datetime_column):
    current_date_in_df = 0
    table_name = f"{database}_{table}"
    if is_questdb_table_exist(table_name) is True:
        query = f"SELECT MAX({datetime_column}) FROM '{table_name}';"
        params = {"query": query}
        response = requests.get(
            config.QUESTDB_CONNECTION_URL, params=params, timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            for val in data["dataset"]:
                if (val is not None) and (val[0] is not None):
                    current_date_in_df = val[0]
                    current_date_in_df = convert_str_to_unix_date(current_date_in_df)
    return current_date_in_df


def get_first_date_from_questdb_df(database, table, datetime_column):
    current_date_in_df = 0
    table_name = f"{database}_{table}"
    if is_questdb_table_exist(table_name) is True:
        query = f"SELECT MIN({datetime_column}) FROM '{table_name}';"
        params = {"query": query}
        response = requests.get(
            config.QUESTDB_CONNECTION_URL, params=params, timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            for val in data["dataset"]:
                if (val is not None) and (val[0] is not None):
                    current_date_in_df = val[0]
                    current_date_in_df = convert_str_to_unix_date(current_date_in_df)

    else:
        raise ValueError(
            f"QuestDB query failed with status code {config.VALUE_ERROR_CODE}"
        )
    return current_date_in_df


def get_latest_date_from_sqldb_df(database, table, datetime_column):
    current_date_in_df = 0
    connection = mysql.connector.connect(
        host=config.IPREDICT_SQL_HOST,
        user=config.IPREDICT_SQL_HOSTNAME,
        passwd=config.IPREDICT_SQL_PASSWD,
    )

    if connection.is_connected() and (
        check_table_exists(connection, database, table) is True
    ):
        logger.debug("Connected to MySQL database")
        cursor = connection.cursor()
        cursor.execute(f"USE {database}")
        query = f"SELECT MAX({datetime_column}) FROM {table};"
        cursor.execute(query)
        result = cursor.fetchone()
        current_date_in_df = result[0]
        if current_date_in_df is not None:
            current_date_in_df = datetime.strptime(
                current_date_in_df, "%Y-%m-%dT%H:%M:%S"
            )
            current_date_in_df = int(current_date_in_df.timestamp())
    return current_date_in_df


def get_latest_date_from_df(database, table, datetime_column):
    database_type = config.IPREDICT_DATABASE_TYPE
    if database_type == "QuestDB":
        return get_latest_date_from_questdb_df(database, table, datetime_column)
    return get_latest_date_from_sqldb_df(database, table, datetime_column)


def resample_data_without_timezone(
    df, resample_unit, agg_func, time_func, output_col=None, label="right"
):
    total_fun = str(time_func) + resample_unit
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.set_index("ds")
    if agg_func == "interpolate":
        logger.debug("Inside interpolate function")
        resampled_df = (
            df.resample(total_fun)
            .apply(lambda x: x.interpolate(method="linear"))
            .reset_index()
        )
    else:
        resampled_df = (
            df.resample(total_fun, label=label)
            .agg({col: agg_func for col in df.columns})
            .reset_index()
        )
    if output_col is None:
        return resampled_df
    return resampled_df[["ds", output_col]]
def get_timezone_offset(timezone_str):
    sign = 1 if timezone_str.startswith("+") else -1
    hours, minutes = map(int, timezone_str[1:].split(":"))
    return tz.tzoffset("local", sign * (hours * 3600 + minutes * 60))
def resample_data(
    df, resample_unit, agg_func, time_func, output_col=None, label="right",
    timezone_str="+00:00", start_time=None, end_time=None,
):
    tz_offset = get_timezone_offset(timezone_str)
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize("UTC").dt.tz_convert(tz_offset)
    df = df.set_index("ds")
    if agg_func == "interpolate":
        resampled_df = (
            df.resample(str(time_func) + resample_unit)
            .apply(lambda x: x.interpolate(method="linear"))
            .reset_index()
        )
    else:
        resampled_df = (
            df.resample(str(time_func) + resample_unit, label=label)
            .agg({col: agg_func for col in df.columns})
            .reset_index()
        )
    resampled_df["start_time"] = resampled_df["ds"]
    if label == "right":
        resampled_df["end_time"] = resampled_df["ds"]
        resampled_df["start_time"] = resampled_df["ds"] - pd.to_timedelta(
            time_func, unit=resample_unit.lower()[0]
        )
    else:
        resampled_df["end_time"] = resampled_df["ds"] + pd.to_timedelta(
            time_func, unit=resample_unit.lower()[0]
        )
    for col in ["ds", "start_time", "end_time"]:
        resampled_df[col] = resampled_df[col].dt.tz_convert("UTC").dt.tz_localize(None)
    user_start = pd.to_datetime(start_time, unit="s") if start_time else None
    user_end = pd.to_datetime(end_time, unit="s") if end_time else None
    if not resampled_df.empty:
        if user_start and resampled_df.iloc[0]["start_time"] < user_start:
            resampled_df.iat[0, resampled_df.columns.get_loc("start_time")] = user_start
        if user_end and resampled_df.iloc[-1]["end_time"] > user_end:
            resampled_df.iat[-1, resampled_df.columns.get_loc("end_time")] = user_end
    columns = ["start_time", "end_time"] + [
        col for col in resampled_df.columns if col not in ["start_time", "end_time"]
    ]
    resampled_df = resampled_df[columns]
    if output_col is None:
        return resampled_df
    return resampled_df[["ds","start_time", "end_time", output_col]]


def delete_table(subject):
    database_list = [
        "anomaly_data",
        "budget_data",
        "cost_data",
        "trend_data",
        "trend_param",
    ]
    trend_list = ["trend", "daily", "weekly"]
    tablename_list = []
    for database in database_list:
        if database != "trend_data":
            table_name = f"{database}_{subject}"
            tablename_list.append(table_name)
        else:
            for trend in trend_list:
                table_name = f"{database}_{subject}_{trend}"
                tablename_list.append(table_name)
    for table in tablename_list:
        if is_questdb_table_exist(table) is True:
            query = f"DROP TABLE '{table}';"
            logger.debug(query)
            params = {"query": query}
            response = requests.get(
                config.QUESTDB_CONNECTION_URL, params=params, timeout=60
            )
            if response.status_code == 200:
                logger.debug("%s deleted", table)
            else:
                logger.debug("unable to delete %s", table)
