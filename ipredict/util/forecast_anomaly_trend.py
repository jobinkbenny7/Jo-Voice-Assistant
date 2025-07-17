# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring,too-many-positional-arguments,too-many-arguments,duplicate-code

import logging
import os
import sys

import pandas as pd

from ipredict.algorithm_impl.anomaly_impl import AnomalyImpl
from ipredict.algorithm_impl.forecasting_impl import ForecastImpl
from ipredict.algorithm_impl.train_inference_impl import TrainInferenceImpl
from ipredict.algorithm_impl.trend_impl import TrendImpl
from ipredict.algorithm_impl.utils import delete_table, save_data_df
from ipredict.config import config

object_train_inference_impl = TrainInferenceImpl()
object_Trend_impl = TrendImpl()
object_Anomaly_impl = AnomalyImpl()
object_Forecast_impl = ForecastImpl()


class ForecastAnomalyTrend:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def train_model_calculate_trend_anomaly(
        self,
        table,
        train_time_type,
        train_time_interval,
        min_val=True,
        lower_val=0.0,
        start_date=None,
        end_date=None,
    ):
        self.logger.debug("function -> train_model_calculate_trend_anomaly")
        input_database = "cost_data"
        analysis_prophet = object_train_inference_impl.get_training_data(
            input_database,
            table,
            train_time_type,
            train_time_interval,
            start_date=start_date,
            end_date=end_date,
        )
        model = object_train_inference_impl.train_model(table, analysis_prophet)
        analysis_fcst = object_train_inference_impl.input_data_prediction(
            model, analysis_prophet
        )
        self.logger.debug("analysis_fcst------------->%s", analysis_fcst.head(5))
        object_Anomaly_impl.data_anomaly(
            table, analysis_fcst, min_val=min_val, lower_val=lower_val
        )
        object_Trend_impl.data_trend(
            table, model, analysis_fcst, min_val=min_val, lower_val=lower_val
        )

    def push_input_data(self, table, input_df):
        self.logger.debug("function -> push_input_data")
        input_database = "cost_data"
        input_df = input_df.reset_index().dropna(subset=["y"])
        save_data_df(input_database, table, input_df, write_mode="append")
        sys.stdout.flush()

    def delete_data(self, subject):
        self.logger.debug("function -> delete_data")
        try:
            delete_table(subject)
        except:
            self.logger.exception("Exception in deleting")

    def train_model_init(
        self,
        table,
        train_time_type,
        train_time_interval,
        input_df,
        min_val=True,
        lower_val=0.0,
        start_date=None,
        end_date=None,
    ):
        self.logger.debug("function -> train_model_init")
        self.push_input_data(table, input_df)
        self.train_model_calculate_trend_anomaly(
            table,
            train_time_type,
            train_time_interval,
            min_val=min_val,
            lower_val=lower_val,
            start_date=start_date,
            end_date=end_date,
        )
        sys.stdout.flush()

    def get_current_state(self, table, context):
        self.logger.debug("function -> get_current_state")
        response = object_train_inference_impl.currentstate(table, context)
        sys.stdout.flush()
        return response

    def fetch_anomaly(
        self,
        table,
        anomaly_filter,
        anomaly_time_type,
        anomaly_time_interval,
        start_time,
        end_time,
        timezone_str,
    ):
        self.logger.debug("function ->  get_anomaly_data")
        response_df = pd.DataFrame()
        try:
            response_df = object_Anomaly_impl.read_anomaly_data_df(
                "anomaly_data",
                table,
                anomaly_filter,
                anomaly_time_type,
                anomaly_time_interval,
                start_time,
                end_time,
                timezone_str,
            )
            sys.stdout.flush()
        except:
            sys.stdout.flush()
        return response_df

    def fetch_anomaly_report(
        self, table, anomaly_time_type, anomaly_time_interval, start_time, end_time
    ):
        self.logger.debug("function ->  fetch_anomaly_report")
        response = {}
        try:
            response = object_Anomaly_impl.anomaly_summary(
                "anomaly_data",
                table,
                anomaly_time_type,
                anomaly_time_interval,
                start_time,
                end_time,
            )
            sys.stdout.flush()
        except:
            sys.stdout.flush()
        return response

    def fetch_trend(
        self,
        table,
        seasonality,
        trend_time_type,
        trend_time_interval,
        start_time,
        end_time,
        timezone_str
    ):
        self.logger.debug("function ->  get_trend_data")
        response_df = pd.DataFrame()
        try:
            response_df = object_Trend_impl.get_trend_data(
                table,
                seasonality,
                trend_time_type,
                trend_time_interval,
                start_time,
                end_time,
                timezone_str,
            )
            sys.stdout.flush()
        except:
            sys.stdout.flush()
        if response_df.empty:
            return pd.DataFrame()
        if 'start_time' in response_df.columns:
            response_df["start_time"] = (
                pd.to_datetime(response_df["start_time"]).astype(int) // 10**9
            )
        if 'end_time' in response_df.columns:
            response_df["end_time"] = (
                pd.to_datetime(response_df["end_time"]).astype(int) // 10**9
            )
        return response_df

    def forecasting(
        self,
        periods,
        frequency_type,
        frequency,
        model_folder_path,
        min_val=True,
        lower_val=0.0,
        start_date=None,
    ):
        response_df = pd.DataFrame()
        try:
            aggregated_forecast_data = os.environ.get(
                "ipredict_aggregated_forecast_data", "True"
            )
            if aggregated_forecast_data == "True":
                response_df = object_Forecast_impl.forecast_agg(
                    periods,
                    frequency_type,
                    frequency,
                    model_folder_path,
                    min_val,
                    lower_val,
                    start_date=start_date,
                )
                response_df["DateTime"] = pd.to_datetime(
                    response_df["DateTime"], errors="coerce"
                )
                freq_map = {
                    "Month": "M",
                    "Week": "W",
                    "Day": "D"
                }
                frequency_type = freq_map.get(frequency_type, frequency_type)
                if frequency_type == "M":
                    response_df["start_time"] = (
                        response_df["DateTime"].dt.to_period("M").dt.start_time
                    )
                    response_df["end_time"] = (
                        response_df["start_time"]
                        + pd.offsets.MonthEnd(0)
                        + pd.Timedelta(hours=23, minutes=59)
                    )
                elif frequency_type == "W":
                    response_df["start_time"] = (
                        response_df["DateTime"]
                        - pd.to_timedelta(
                            response_df["DateTime"].dt.weekday + 1, unit="D"
                        )
                        + pd.to_timedelta(1, unit="D")
                    )
                    response_df["end_time"] = response_df["start_time"] + pd.Timedelta(
                        days=6, hours=23, minutes=59
                    )
                elif frequency_type == "D":
                    response_df["start_time"] = response_df["DateTime"].dt.normalize()
                    response_df["end_time"] = response_df["start_time"] + pd.Timedelta(
                        hours=23, minutes=59
                    )
                else:
                    raise ValueError(f"Unsupported frequency_type: {frequency_type}")
                response_df.loc[1:, "start_time"] = response_df["end_time"].shift(1)[1:]
                response_df = response_df.rename(
                    columns={
                        "yhat": "Expected",
                        "yhat_upper": "Expected_max",
                        "yhat_lower": "Expected_min",
                    }
                )
            else:
                response_df = object_Forecast_impl.forecasting_for_specific_time(
                    periods,
                    frequency_type,
                    frequency,
                    model_folder_path,
                    min_val,
                    lower_val,
                )
            response_df["start_time"] = (
                pd.to_datetime(response_df["start_time"], errors="coerce").astype(int)
                // 10**9
            )
            response_df["end_time"] = (
                pd.to_datetime(response_df["end_time"], errors="coerce").astype(int)
                // 10**9
            )

        except Exception as e:
            self.logger.error("Error in forecasting: %s", str(e))
            raise
        return response_df[
            ["start_time", "end_time", "Expected", "Expected_max", "Expected_min"]
        ]

    def calculate_budget(self, model_folder_path, start_year, month, year_count):
        self.logger.debug("function -> calculate_budget")
        try:
            object_Forecast_impl.monthly_budget_forecast(
                model_folder_path, start_year, month, year_count
            )
            sys.stdout.flush()
        except:
            sys.stdout.flush()
            sys.exit(config.VALUE_ERROR_CODE)

    def get_budget(self, table):
        self.logger.debug("function -> get_budget")
        response_df = pd.DataFrame()
        try:
            response_df = object_Forecast_impl.budget_data(table)
            sys.stdout.flush()
        except:
            sys.stdout.flush()
        if 'DateTime' in response_df.columns:
            response_df["DateTime"] = (
            pd.to_datetime(response_df["DateTime"]).astype(int) // 10**9
        )
        return response_df

    def get_new_budget(self, parent_table, effecting_table):
        self.logger.debug("function -> get_new_budget")
        response_df = pd.DataFrame()
        try:
            response_df = object_Forecast_impl.budget_diff(
                parent_table, effecting_table
            )
            sys.stdout.flush()
        except:
            sys.stdout.flush()
        if 'DateTime' in response_df.columns:
            response_df["DateTime"] = (
                pd.to_datetime(response_df["DateTime"]).astype(int) // 10**9
            )
        return response_df

    def get_price_change(self, table, month, year, original_price, changed_price):
        self.logger.debug("get_price_change")
        response_df = pd.DataFrame()
        try:
            response_df = object_Forecast_impl.new_price_calculation(
                table, month, year, original_price, changed_price
            )
            sys.stdout.flush()
        except:
            sys.stdout.flush()
        if 'DateTime' in response_df.columns:
            response_df["DateTime"] = (
            pd.to_datetime(response_df["DateTime"]).astype(int) // 10**9
        )
        return response_df
