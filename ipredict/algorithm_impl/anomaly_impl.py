# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring,too-many-positional-arguments,too-many-arguments,too-many-locals

import logging
import os
from datetime import date, datetime

import pandas as pd

from ipredict.algorithm_impl.utils import (
    FREQ_DICT, execute_query_get_df, get_questdb_data_query,
    get_questdb_data_query_with_start_end_date_mandatory,
    get_resample_start_end_date_from_input, resample_data,
    resample_data_without_timezone, save_data_df)


class AnomalyImpl:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def data_anomaly(self, table, analysis_fcst, min_val=False, lower_val=0.0):
        self.logger.debug("function ->  data_anomaly")
        output_database = "anomaly_data"
        if analysis_fcst.shape[0] != 0:
            analysis_st_df = analysis_fcst[
                ["ds", "yhat", "yhat_lower", "yhat_upper", "fact"]
            ].copy()
            if min_val is True:
                try:
                    analysis_st_df.loc[:, "yhat"] = analysis_st_df["yhat"].clip(
                        lower=lower_val
                    )
                    analysis_st_df.loc[:, "yhat_lower"] = analysis_st_df[
                        "yhat_lower"
                    ].clip(lower=lower_val)
                    analysis_st_df.loc[:, "yhat_upper"] = analysis_st_df[
                        "yhat_upper"
                    ].clip(lower=lower_val)
                except Exception:
                    analysis_st_df.loc["yhat"] = analysis_st_df["yhat"].clip(
                        lower=lower_val
                    )
                    analysis_st_df.loc["yhat_lower"] = analysis_st_df[
                        "yhat_lower"
                    ].clip(lower=lower_val)
                    analysis_st_df.loc["yhat_upper"] = analysis_st_df[
                        "yhat_upper"
                    ].clip(lower=lower_val)

            analysis_st_df["Anomaly"] = "Normal"
            analysis_st_df.loc[
                analysis_st_df["fact"] > analysis_st_df["yhat_upper"], "Anomaly"
            ] = "Anomaly"
            analysis_st_df.loc[
                analysis_st_df["fact"] < analysis_st_df["yhat_lower"], "Anomaly"
            ] = "Anomaly"

            analysis_st_df["Deviation"] = (
                analysis_st_df["fact"] - analysis_st_df["yhat"]
            )
            analysis_st_df = analysis_st_df.reset_index().rename(
                columns={
                    "yhat": "Expected",
                    "yhat_upper": "Expected Maximum",
                    "yhat_lower": "Expected Minimum",
                    "fact": "Current Value",
                }
            )
            save_data_df(output_database, table, analysis_st_df)

    def anomaly_agg(self, anomaly_df, anomaly_time_type, anomaly_time_interval, timezone_str,
                     start_time, end_time):
        if (start_time is None) and (end_time is None):
            return anomaly_df
        resample_df = anomaly_df[
            [
                "ds",
                "Expected",
                "Expected Minimum",
                "Expected Maximum",
                "Current Value",
                "Deviation",
            ]
        ]
        resample_df = resample_df[resample_df["ds"] >= start_time]
        resample_df = resample_df[resample_df["ds"] < end_time]
        resampled_df = resample_data(
            resample_df,
            anomaly_time_type,
            "sum",
            anomaly_time_interval,
            label="right",
            timezone_str=timezone_str,
            start_time=start_time,
            end_time=end_time,
        )
        resampled_df["Anomaly"] = "Normal"
        resampled_df.loc[
            resampled_df["Current Value"] > resampled_df["Expected Maximum"],
            "Anomaly",
        ] = "Anomaly"
        resampled_df.loc[
            resampled_df["Current Value"] < resampled_df["Expected Minimum"],
            "Anomaly",
        ] = "Anomaly"
        return resampled_df
    def anomaly_agg_report(self, anomaly_df, anomaly_time_type, anomaly_time_interval,
                           start_time, end_time):
        start_time = list(anomaly_df["ds"])[0]
        end_time = list(anomaly_df["ds"])[-1]
        start_time, end_time = get_resample_start_end_date_from_input(
            anomaly_df, anomaly_time_type, anomaly_time_interval
        )
        if (start_time is None) and (end_time is None):
            return anomaly_df

        resample_df = anomaly_df[
            [
                "ds",
                "Expected",
                "Expected Minimum",
                "Expected Maximum",
                "Current Value",
                "Deviation",
            ]
        ]
        resample_df = resample_df[resample_df["ds"] >= start_time]
        resample_df = resample_df[resample_df["ds"] < end_time]
        resampled_df = resample_data_without_timezone(
            resample_df,
            anomaly_time_type,
            "sum",
            anomaly_time_interval,
            label="right",
        )
        resampled_df["Anomaly"] = "Normal"
        resampled_df.loc[
            resampled_df["Current Value"] > resampled_df["Expected Maximum"],
            "Anomaly",
        ] = "Anomaly"
        resampled_df.loc[
            resampled_df["Current Value"] < resampled_df["Expected Minimum"],
            "Anomaly",
        ] = "Anomaly"
        return resampled_df

    def read_anomaly_questdb_data_df(
        self,
        database,
        table,
        anomaly_filter,
        anomaly_time_type,
        anomaly_time_interval,
        start_date,
        end_date,
        timezone_str,
    ):
        anomaly_start_end_date_mandatory = os.environ.get(
            "ipredict_anomaly_start_end_date_mandatory", "True"
        )
        self.logger.debug(
            "start end date mandtory -> %s", anomaly_start_end_date_mandatory
        )
        if (
            (anomaly_start_end_date_mandatory != "True")
            or (start_date is None)
            or (end_date is None)
        ):
            query = get_questdb_data_query(database, table, start_date, end_date)
        else:
            query = get_questdb_data_query_with_start_end_date_mandatory(
                database, table, start_date, end_date
            )

        if anomaly_filter != "All":
            if "WHERE" in query:
                query = query + f" AND Anomaly = '{anomaly_filter}'"
            else:
                query = query + f" WHERE Anomaly = '{anomaly_filter}'"
        anomaly_df = execute_query_get_df(query)
        if anomaly_df is None or anomaly_df.empty:
            return pd.DataFrame()
        anomaly_df["ds"] = pd.to_datetime(anomaly_df["ds"])
        if (anomaly_time_type != "Input") and (anomaly_df.shape[0] != 0):
            anomaly_df = self.anomaly_agg(
                anomaly_df, anomaly_time_type, anomaly_time_interval, timezone_str, start_date,
                end_date
            )
        anomaly_df["start_time"] = pd.to_datetime(anomaly_df["start_time"]).astype(int) // 10**9
        anomaly_df["end_time"] = pd.to_datetime(anomaly_df["end_time"]).astype(int) // 10**9
        return anomaly_df
    def read_anomaly_questdb_data_df_report(
        self,
        database,
        table,
        anomaly_filter,
        anomaly_time_type,
        anomaly_time_interval,
        start_date,
        end_date,
    ):
        anomaly_start_end_date_mandatory = os.environ.get(
            "ipredict_anomaly_start_end_date_mandatory", "True"
        )
        self.logger.debug(
            "start end date mandtory -> %s", anomaly_start_end_date_mandatory
        )
        if (
            (anomaly_start_end_date_mandatory != "True")
            or (start_date is None)
            or (end_date is None)
        ):
            query = get_questdb_data_query(database, table, start_date, end_date)
        else:
            query = get_questdb_data_query_with_start_end_date_mandatory(
                database, table, start_date, end_date
            )

        if anomaly_filter != "All":
            if "WHERE" in query:
                query = query + f" AND Anomaly = '{anomaly_filter}'"
            else:
                query = query + f" WHERE Anomaly = '{anomaly_filter}'"
        anomaly_df = execute_query_get_df(query)
        if anomaly_df is None or anomaly_df.empty:
            return pd.DataFrame()
        anomaly_df["ds"] = pd.to_datetime(anomaly_df["ds"])
        if (anomaly_time_type != "Input") and (anomaly_df.shape[0] != 0):
            anomaly_df = self.anomaly_agg_report(
                anomaly_df, anomaly_time_type, anomaly_time_interval, start_date,
                end_date
            )
        anomaly_df["ds"] = pd.to_datetime(anomaly_df["ds"]).astype(int) // 10**9
        return anomaly_df

    def read_anomaly_data_df(
        self,
        database,
        table,
        anomaly_filter,
        anomaly_time_type,
        anomaly_time_interval,
        start_date=None,
        end_date=None,
        timezone_str=None,
    ):
        database_type = os.environ.get("ipredict_database_type", "QuestDB")
        freq_dict = FREQ_DICT
        if anomaly_time_type != "Input":
            anomaly_time_type = freq_dict[anomaly_time_type]
        if database_type == "QuestDB":
            return self.read_anomaly_questdb_data_df(
                database,
                table,
                anomaly_filter,
                anomaly_time_type,
                anomaly_time_interval,
                start_date,
                end_date,
                timezone_str=timezone_str,
            )
        raise ValueError(f"Unsupported database type: {database_type}")

    def anomaly_summary(
        self,
        database,
        table,
        anomaly_time_type,
        anomaly_time_interval,
        start_date,
        end_date,
    ):
        freq_dict = FREQ_DICT
        anomaly_filter = "All"
        if anomaly_time_type != "Input":
            anomaly_time_type = freq_dict[anomaly_time_type]
        total_df = self.read_anomaly_questdb_data_df_report(
            database,
            table,
            anomaly_filter,
            anomaly_time_type,
            anomaly_time_interval,
            start_date,
            end_date,
        )
        if total_df.empty:
            return pd.DataFrame()
        total_data = total_df.shape[0]
        num_anomalies = 0
        anomaly_percentage = 0
        anomaly_counts_by_date = 0
        max_anomaly_date = 0
        if total_data != 0:
            num_anomalies = (total_df["Anomaly"] == "Anomaly").sum()
            if num_anomalies != 0:
                anomaly_percentage = (num_anomalies / total_data) * 100
                total_df["ds"] = pd.to_datetime(total_df["ds"], unit="s")
                anomaly_counts_by_date = (
                    total_df[total_df["Anomaly"] == "Anomaly"].groupby("ds").size()
                )
                max_anomaly_date = anomaly_counts_by_date.idxmax()
                dt_date = date(
                    max_anomaly_date.year, max_anomaly_date.month, max_anomaly_date.day
                )

                dt_datetime = datetime.combine(dt_date, datetime.min.time())
                max_anomaly_date = int(dt_datetime.timestamp())

        self.logger.info("Number of anomalies: %s", num_anomalies)
        self.logger.info("Percentage of anomalies: %.2f%%", anomaly_percentage)
        self.logger.info("Date with maximum anomalies: %s", max_anomaly_date)

        response = {
            "num_anomalies": num_anomalies,
            "Anomaly_percentage": anomaly_percentage,
            "max_anomaly_date": max_anomaly_date,
        }
        return response
