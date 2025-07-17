# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring,too-many-positional-arguments,too-many-arguments

import json
import logging

import grpc
import pandas as pd
from prophet import Prophet

from ipredict.algorithm_impl.utils import (
    FREQ_DICT, get_data_time_diff_in_seconds, get_first_date_from_questdb_df,
    get_latest_date_from_df, get_resample_start_end_date_from_input,
    is_table_exists, read_data_df, read_model, resample_data, save_model)


class TrainInferenceImpl:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def get_resampled_training_data(
        self, train_data_df, train_time_type, train_time_interval
    ):
        freq_dict = FREQ_DICT
        train_time_type = freq_dict[train_time_type]
        start_time, end_time = get_resample_start_end_date_from_input(
            train_data_df, train_time_type, train_time_interval
        )
        if (start_time is None) and (end_time is None):
            return train_data_df

        resample_df = train_data_df[train_data_df["ds"] >= start_time]
        resample_df = resample_df[resample_df["ds"] < end_time]
        resampled_df = resample_data(
            resample_df, train_time_type, "sum", train_time_interval, label="right"
        )
        self.logger.debug("Resampled data shape -> %s", resampled_df.shape)
        return resampled_df

    def get_training_data(
        self,
        database,
        table,
        train_time_type,
        train_time_interval,
        start_date=None,
        end_date=None,
    ):
        self.logger.debug("database -> %s, table -> %s", database, table)
        train_data_df = read_data_df(
            database, table, start_date=start_date, end_date=end_date
        )
        train_data_df = train_data_df[["ds", "y"]]
        train_data_df["ds"] = pd.to_datetime(train_data_df["ds"])
        if train_time_type != "Input":
            train_data_df = self.get_resampled_training_data(
                train_data_df, train_time_type, train_time_interval
            )
        train_data_df = train_data_df.set_index("ds")
        train_data_df = train_data_df.reset_index().dropna(subset=["y"])

        return train_data_df

    def input_data_prediction(self, model, analysis_prophet):
        self.logger.debug("function ->   data_prediction")
        analysis_fcst = []
        analysis_prophet["ds"] = pd.to_datetime(analysis_prophet["ds"])
        grouped = analysis_prophet.groupby(analysis_prophet["ds"].dt.date)
        for _, group in grouped:
            forecast = model.predict(group)
            forecast["fact"] = group["y"].values
            analysis_fcst.append(forecast)
        if analysis_fcst:
            return pd.concat(analysis_fcst, ignore_index=True)
        return pd.DataFrame()

    def train_model(self, table, analysis_prophet):
        self.logger.debug("function ->  train model")
        if analysis_prophet.shape[0] != 0:
            model = Prophet(
                growth="linear",
                seasonality_mode="additive",
                changepoint_prior_scale=0.5,
                holidays_prior_scale=0.01,
                interval_width=0.9,
                seasonality_prior_scale=10,
                daily_seasonality=True,
                mcmc_samples=0,
            )
            model.fit(analysis_prophet)
            save_model(model, table)
            return model
        return None

    def currentstate(self, subject, context):
        model_start_time = 0
        model_end_time = 0
        input_data_end_time = 0
        input_data_start_time = 0
        input_data_time_diff_in_sec = 0
        model_exist = False
        budget_calculated = False

        model = read_model(subject)
        if model is None:
            self.logger.error("Model could not be read.")
            model_exist = False
        else:
            model_exist = True
            df = model.history
            model_start_time = df["ds"].min().timestamp()
            model_end_time = df["ds"].max().timestamp()
            dt = model.history["ds"].diff()
            input_data_time_diff_in_sec = get_data_time_diff_in_seconds(dt)
        if not is_table_exists(f"cost_data_{subject}"):
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(f"Table cost_data_{subject} is not available in QuestDB.")
            return None
        if is_table_exists(f"cost_data_{subject}") is True:
            input_database = "cost_data"
            input_data_end_time = get_latest_date_from_df(
                input_database, subject, datetime_column="timestamp"
            )
            input_data_start_time = get_first_date_from_questdb_df(
                input_database, subject, datetime_column="timestamp"
            )

        budget_calculated = bool(is_table_exists(f"budget_{subject}"))

        response = {
            "model_exist": model_exist,
            "model_start_time": int(model_start_time),
            "model_end_time": int(model_end_time),
            "input_data_time_diff_in_sec": int(input_data_time_diff_in_sec),
            "input_data_end_time": int(input_data_end_time),
            "input_data_start_time": int(input_data_start_time),
            "budget_calculated": budget_calculated,
        }

        self.logger.info(json.dumps(response, indent=4))
        return response
