# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring,too-many-positional-arguments,too-many-arguments,too-many-locals,too-many-statements,consider-using-from-import

import logging
import math
import os
from datetime import datetime, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta

import ipredict.config.config as config
from ipredict.algorithm_impl.utils import (FREQ_DICT, convert_str_to_unix_date,
                                           convert_unix_to_str_date,
                                           get_data_time_diff_in_seconds,
                                           get_latest_date_from_df,
                                           read_data_df, read_model,
                                           resample_data_without_timezone, save_data_df)


class ForecastImpl:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def forecasting_for_specific_time(
        self,
        periods,
        frequency_type,
        frequency,
        model_folder_path,
        min_val=False,
        lower_val=0.0,
    ):
        self.logger.debug("function ->  forecasting")
        freq_dict = FREQ_DICT
        freq_str = str(frequency) + freq_dict[frequency_type]
        model = read_model(model_folder_path)
        if model is not None:
            future_data = model.make_future_dataframe(
                periods=periods, freq=freq_str, include_history=False
            )
            future_data_fcst = model.predict(future_data)
            if min_val is True:
                try:
                    future_data_fcst["yhat"] = future_data_fcst["yhat"].clip(
                        lower=lower_val
                    )
                    future_data_fcst["yhat_lower"] = future_data_fcst[
                        "yhat_lower"
                    ].clip(lower=lower_val)
                    future_data_fcst["yhat_upper"] = future_data_fcst[
                        "yhat_upper"
                    ].clip(lower=lower_val)
                except KeyError:
                    future_data_fcst[:, "yhat"] = future_data_fcst["yhat"].clip(
                        lower=lower_val
                    )
                    future_data_fcst[:, "yhat_lower"] = future_data_fcst[
                        "yhat_lower"
                    ].clip(lower=lower_val)
                    future_data_fcst[:, "yhat_upper"] = future_data_fcst[
                        "yhat_upper"
                    ].clip(lower=lower_val)
            future_data_df = future_data_fcst.reset_index().rename(
                columns={
                    "ds": "DateTime",
                    "yhat": "Expected",
                    "yhat_upper": "Expected_max",
                    "yhat_lower": "Expected_min",
                }
            )
            future_data_df = future_data_df[
                ["DateTime", "Expected", "Expected_max", "Expected_min"]
            ]
            return future_data_df
        return None

    def get_end_time_for_forcast(self, start_time, frequency_type, periods, frequency):

        if frequency_type == "M":
            end_time = start_time + relativedelta(months=periods * frequency)
        elif frequency_type == "W":
            end_time = start_time + timedelta(weeks=periods * frequency)
        elif frequency_type == "D":
            end_time = start_time + timedelta(days=periods * frequency)
        elif frequency_type == "H":
            end_time = start_time + timedelta(hours=periods * frequency)
        elif frequency_type == "min":
            end_time = start_time + timedelta(minutes=periods * frequency)

        if frequency_type == "M":
            if frequency > 1:
                end_time = end_time + relativedelta(months=1)
        elif frequency_type == "W":
            if frequency > 1:
                end_time = end_time + timedelta(days=7)

        return end_time

    def get_start_time_for_forecast(self, frequency_type, start_date=None):
        if start_date is None:
            start_date = datetime.today()
            self.logger.debug("start_date -> datetime.today()  --> %s", start_date)
        else:
            try:
                start_date = convert_unix_to_str_date(start_date, time_zone=False)
            except ValueError:
                start_date = datetime.today()
        if frequency_type == "M":
            start_date = datetime(start_date.year, start_date.month, 1)
        elif frequency_type == "W":
            week_day_offset = (start_date.weekday() + 1) % 7
            start_date = (start_date - timedelta(days=week_day_offset - 1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif frequency_type == "D":
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif frequency_type == "H":
            start_date = start_date.replace(minute=0, second=0, microsecond=0)
        elif frequency_type == "min":
            start_date = start_date.replace(second=0, microsecond=0)
        return start_date

    def get_start_time_for_forecast_using_resample(
        self, frequency_type, frequency, start_date
    ):
        try:
            if start_date is None:
                start_date = datetime.today()
                self.logger.debug("start_date -> datetime.today()  --> %s", start_date)
            else:
                start_date = convert_unix_to_str_date(start_date, time_zone=False)
            dummy_df = pd.DataFrame()
            dummy_df["ds"] = [start_date]
            dummy_df["yhat"] = [2]
            resample_df = resample_data_without_timezone(
                dummy_df,
                frequency_type,
                "sum",
                frequency,
                output_col="yhat",
                label="left",
            )

            start_date = list(resample_df["ds"])[0]
            if frequency_type == "M":
                start_date = start_date + timedelta(days=1)
                start_date = datetime(start_date.year, start_date.month, 1)
                if frequency > 1:
                    start_date = start_date + relativedelta(months=frequency - 2)
            elif frequency_type == "W":
                week_day_offset = (start_date.weekday() + 1) % 7
                start_date = (start_date - timedelta(days=week_day_offset - 1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                if frequency > 1:
                    start_date = start_date + timedelta(days=7 * (frequency - 2))
            return start_date
        except Exception:
            self.logger.exception(
                "Inside exception so calling get_start_time_for_forecast "
            )
            return self.get_start_time_for_forecast(frequency_type)

    def forecast_agg(
        self,
        periods,
        frequency_type,
        frequency,
        model_folder_path,
        min_val=True,
        lower_val=0.0,
        start_date=None,
        label="right",
    ):
        freq_dict = FREQ_DICT
        frequency_type = freq_dict[frequency_type]

        model = read_model(model_folder_path)
        if model is None:
            self.logger.error("Model not found.")
            raise ValueError(
                f"QuestDB query failed with status code {config.VALUE_ERROR_CODE}"
            )

        dt = model.history["ds"].diff()
        input_data_time_diff_in_sec = get_data_time_diff_in_seconds(dt)
        self.logger.debug(
            "input_data_time_diff_in_sec --> %s", input_data_time_diff_in_sec
        )
        start_time = self.get_start_time_for_forecast_using_resample(
            frequency_type, frequency, start_date
        )
        end_time = self.get_end_time_for_forcast(
            start_time, frequency_type, periods, frequency
        )
        self.logger.info("Start time: %s, End time: %s", start_time, end_time)

        start_end_seconds = (end_time - start_time).total_seconds()
        self.logger.debug("Time difference: %s seconds", start_end_seconds)

        plot_points = int(math.ceil(start_end_seconds / input_data_time_diff_in_sec))
        future = pd.DataFrame(
            {
                "ds": [
                    start_time + timedelta(seconds=input_data_time_diff_in_sec * i)
                    for i in range(plot_points)
                ]
            }
        )
        agg_func = "sum"
        if frequency_type in ["M", "W", "D"]:
            self.logger.debug(
                "Prediction started for frequency type %s", frequency_type
            )
            forecast_list = []

            future["year_month"] = future["ds"].dt.to_period(frequency_type)
            grouped = future.groupby("year_month")

            for period, group in grouped:

                forecast_chunk = model.predict(group)
                if not forecast_chunk.empty:

                    aggregated = forecast_chunk[
                        ["yhat", "yhat_lower", "yhat_upper"]
                    ].sum()

                    aggregated["year"] = period.year
                    aggregated["month"] = (
                        period.month if frequency_type == "M" else None
                    )

                    aggregated["ds"] = period.end_time.strftime("%Y-%m-%d")
                    forecast_list.append(aggregated)
                else:
                    self.logger.warning("No forecast data for month: %s", period)
            if forecast_list:
                forecast = pd.DataFrame(forecast_list)
                self.logger.debug(
                    "Prediction completed. Forecast shape: %s", forecast.shape
                )
            else:
                self.logger.error("No forecast data generated.")
                forecast = pd.DataFrame()

            threshold = float(
                os.environ.get(
                    "IPREDICT_FORECAST_THRESHOLD", config.IPREDICT_FORECAST_THRESHOLD
                )
            )
            self.logger.debug("Forecasting threshold -> %s", threshold)

            if min_val:
                forecast["yhat_lower"] = forecast["yhat_lower"].where(
                    forecast["yhat_lower"]
                    > forecast["yhat"] - threshold * forecast["yhat"],
                    forecast["yhat"] - threshold * forecast["yhat"],
                )
                forecast["yhat_upper"] = forecast["yhat_upper"].where(
                    forecast["yhat_upper"]
                    < forecast["yhat"] + threshold * forecast["yhat"],
                    forecast["yhat"] + threshold * forecast["yhat"],
                )
                forecast["yhat"] = forecast["yhat"].clip(lower=lower_val)
                forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=lower_val)
                forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=lower_val)

            forecast = forecast[["ds", "yhat", "yhat_upper", "yhat_lower"]]
            forecast = forecast.reset_index().rename(columns={"ds": "DateTime"})
            self.logger.debug("Output df:\n%s", forecast.head())
            forecast = forecast.rename(
                columns={
                    "yhat": "Expected",
                    "yhat_upper": "Expected_max",
                    "yhat_lower": "Expected_min",
                }
            )
            forecast["DateTime"] = pd.to_datetime(forecast["DateTime"], errors="coerce")
            return forecast

        self.logger.debug("Prediction started")
        forecast = model.predict(future)
        self.logger.debug("Prediction completed. Forecast shape: %s", forecast.shape)

        threshold = float(
            os.environ.get(
                "IPREDICT_FORECAST_THRESHOLD", config.IPREDICT_FORECAST_THRESHOLD
            )
        )
        self.logger.debug("Forecasting threshold -> %s", threshold)
        if min_val is True:
            try:
                forecast["yhat_lower"] = forecast["yhat_lower"].where(
                    forecast["yhat_lower"]
                    > forecast["yhat"] - threshold * forecast["yhat"],
                    forecast["yhat"] - threshold * forecast["yhat"],
                )

                forecast["yhat_upper"] = forecast["yhat_upper"].where(
                    forecast["yhat_upper"]
                    < forecast["yhat"] + threshold * forecast["yhat"],
                    forecast["yhat"] + threshold * forecast["yhat"],
                )

                forecast["yhat"] = forecast["yhat"].clip(lower=lower_val)
                forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=lower_val)
                forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=lower_val)

            except Exception:
                forecast[:, "yhat"] = forecast["yhat"].clip(lower=lower_val)
                forecast[:, "yhat_lower"] = forecast["yhat_lower"].clip(lower=lower_val)
                forecast[:, "yhat_upper"] = forecast["yhat_upper"].clip(lower=lower_val)
        future_data_df = forecast.reset_index().rename(
            columns={
                "ds": "ds",
                "yhat": "Expected",
                "yhat_upper": "Expected_max",
                "yhat_lower": "Expected_min",
            }
        )
        output_df = future_data_df[["ds", "Expected", "Expected_max", "Expected_min"]]
        forecast_replace_expected_with_actual_if_exist = os.environ.get(
            "ipredict_forecast_replace_expected_with_actual_if_exist", "True"
        )
        if forecast_replace_expected_with_actual_if_exist == "True":
            input_database = "cost_data"
            latest_timestamp = get_latest_date_from_df(
                input_database, table=model_folder_path, datetime_column="timestamp"
            )
            self.logger.info("latest time stamp -> %s", latest_timestamp)
            time_format = "%Y-%m-%d %H:%M:%S"
            start_time_str = start_time.strftime(time_format)
            unix_start_time = convert_str_to_unix_date(
                start_time_str, format=time_format
            )
            if unix_start_time < latest_timestamp:
                self.logger.debug(
                    "latest input timestamp -> %s "
                    "and forecast_start_time_satmp -> %s",
                    latest_timestamp,
                    unix_start_time,
                )
                replace_input_df = read_data_df(
                    "cost_data",
                    table=model_folder_path,
                    start_date=unix_start_time,
                    end_date=latest_timestamp,
                )
                replace_input_df = replace_input_df[["ds", "y"]]
                output_df = pd.merge(output_df, replace_input_df, on="ds", how="left")
                output_df.loc[(output_df["y"].notna()), "Expected"] = output_df["y"]
                output_df = output_df[
                    ["ds", "Expected", "Expected_max", "Expected_min"]
                ]
        output_data_time_diff_in_sec = get_data_time_diff_in_seconds(
            output_df["ds"].diff()
        )
        self.logger.info(
            "output_data_time_diff_in_sec -> %s", output_data_time_diff_in_sec
        )
        resampled_df = resample_data_without_timezone(
            output_df, frequency_type, agg_func, frequency, label=label
        )
        resampled_df = resampled_df.reset_index().rename(columns={"ds": "DateTime"})
        resampled_df = resampled_df[
            ["DateTime", "Expected", "Expected_max", "Expected_min"]
        ]
        return resampled_df

    def yearly_budget_forecast(self, model_folder_path, start_year, month, year_count):
        self.logger.info(start_year)
        all_forecasts = pd.DataFrame()
        no_of_quarters_per_year = 4
        quarter_length = 3
        for i in range(year_count):
            start_date = f"{start_year+i}-{month}-01 00:00:00"
            start_time_unix = convert_str_to_unix_date(
                start_date, format="%Y-%m-%d %H:%M:%S"
            )
            self.logger.info("start_time_unix-----> %s", start_time_unix)
            resample_df = self.forecast_agg(
                periods=no_of_quarters_per_year,
                frequency_type="Month",
                frequency=quarter_length,
                model_folder_path=model_folder_path,
                start_date=start_time_unix,
                min_val=True,
                lower_val=0.0,
                label="right",
            )
            resample_df["QHY"] = [f"Q{i + 1}" for i in range(no_of_quarters_per_year)]
            col_list = ["Expected", "Expected_max", "Expected_min"]
            rows = []
            date_time_list = list(resample_df["DateTime"])
            for j in range(int(resample_df.shape[0] / 2)):
                h = {}
                for col in col_list:
                    h[col] = resample_df.loc[(j * 2) : (j * 2) + 1, col].sum()
                rows.append(h)
                rows[j]["DateTime"] = date_time_list[(j * 2) + 1]
                rows[j]["QHY"] = f"H{j+1}"
            y = {}
            for col in col_list:
                y[col] = rows[0][col] + rows[1][col]
            y["DateTime"] = rows[1]["DateTime"]
            y["QHY"] = "YY"
            rows.append(y)
            df = pd.DataFrame(rows)
            resample_df = pd.concat([resample_df, df], ignore_index=True)
            resample_df["Year"] = [
                (start_year + i) for h in range(resample_df.shape[0])
            ]
            all_forecasts = pd.concat([all_forecasts, resample_df], ignore_index=True)
        self.logger.debug(
            "Budget for %s years start from %s ->\n%s",
            year_count,
            start_year,
            all_forecasts,
        )
        save_data_df(
            "budget",
            table=model_folder_path,
            data_df=all_forecasts,
            write_mode="replace",
        )

    def monthly_budget_forecast(self, model_folder_path, start_year, month, year_count):
        all_forecasts = pd.DataFrame()
        months_per_year = 12

        for i in range(year_count):
            start_date = f"{start_year+i}-{month}-01 00:00:00"
            start_time_unix = convert_str_to_unix_date(
                start_date, format="%Y-%m-%d %H:%M:%S"
            )
            self.logger.debug("start_time_unix-----> %s", start_time_unix)

            resample_df = self.forecast_agg(
                periods=months_per_year,
                frequency_type="Month",
                frequency=1,
                model_folder_path=model_folder_path,
                start_date=start_time_unix,
                min_val=True,
                lower_val=0.0,
                label="right",
            )

            if resample_df.shape[0] != months_per_year:
                self.logger.error(
                    "Expected %s rows for monthly data, got %s.",
                    months_per_year,
                    resample_df.shape[0],
                )
                continue

            resample_df["QHY"] = [f"M{j+1}" for j in range(months_per_year)]
            col_list = ["Expected", "Expected_max", "Expected_min"]

            rows = []
            date_time_list = list(resample_df["DateTime"])
            for j in range(int(resample_df.shape[0] / 3)):
                q = {}
                for col in col_list:
                    q[col] = resample_df.loc[(j * 3) : (j * 3) + 2, col].sum()
                rows.append(q)
                rows[j]["DateTime"] = date_time_list[(j * 3) + 2]
                rows[j]["QHY"] = f"Q{j+1}"

            h_rows = []
            for j in range(int(len(rows) / 2)):
                h = {}
                for col in col_list:
                    h[col] = sum([rows[j * 2][col], rows[j * 2 + 1][col]])
                h_rows.append(h)
                h_rows[j]["DateTime"] = rows[(j * 2) + 1]["DateTime"]
                h_rows[j]["QHY"] = f"H{j+1}"

            y = {}
            for col in col_list:
                y[col] = h_rows[0][col] + h_rows[1][col]
            y["DateTime"] = h_rows[1]["DateTime"]
            y["QHY"] = "YY"
            h_rows.append(y)

            qh_df = pd.DataFrame(rows)
            hy_df = pd.DataFrame(h_rows)
            combined_df = pd.concat([resample_df, qh_df, hy_df], ignore_index=True)
            combined_df["Year"] = [
                (start_year + i) for _ in range(combined_df.shape[0])
            ]
            all_forecasts = pd.concat([all_forecasts, combined_df], ignore_index=True)

        self.logger.debug(
            "Budget for %s years start from %s ->\n%s",
            year_count,
            start_year,
            all_forecasts,
        )
        save_data_df(
            "budget",
            table=model_folder_path,
            data_df=all_forecasts,
            write_mode="replace",
        )

    def budget_data(self, table):
        budget_data_df = read_data_df("budget", table=table)
        budget_data_df = budget_data_df.reset_index().rename(columns={"ds": "DateTime"})
        budget_data_df = budget_data_df[
            ["DateTime", "Year", "QHY", "Expected", "Expected_max", "Expected_min"]
        ]
        return budget_data_df

    def budget_diff(self, parent_table, effecting_table):
        budget_parent_table = self.budget_data(parent_table)
        self.logger.debug(budget_parent_table)
        budget_effecting_table = self.budget_data(effecting_table)
        self.logger.debug(budget_effecting_table)
        new_budget = pd.DataFrame()
        assign_col_list = ["DateTime", "QHY", "Year"]
        for col in assign_col_list:
            new_budget[col] = budget_parent_table[col]
        diff_col_list = ["Expected", "Expected_max", "Expected_min"]
        for col in diff_col_list:
            new_budget[col] = budget_parent_table[col] - budget_effecting_table[col]
        self.logger.debug(new_budget)
        return new_budget

    def price_calculation(
        self, table, original_price, changed_price, modified_table_name
    ):
        budget_df = self.budget_data(table)
        col_list = ["Expected", "Expected_max", "Expected_min"]
        for col in col_list:
            budget_df[col] = (budget_df[col] / original_price) * changed_price
        save_data_df(
            "budget", table=modified_table_name, data_df=budget_df, write_mode="replace"
        )
        return budget_df

    def new_price_calculation(self, table, month, year, original_price, changed_price):
        start_date = f"{year}-{month}-01 00:00:00"
        start_time_unix = convert_str_to_unix_date(
            start_date, format="%Y-%m-%d %H:%M:%S"
        )
        self.logger.info("start_time_unix-----> %s", start_time_unix)
        budget_df = self.budget_data(table)
        self.logger.debug("Starting Budget")
        self.logger.debug(budget_df)
        budget_df["DateTime"] = pd.to_datetime(budget_df["DateTime"])
        start_time_datetime = pd.to_datetime(start_time_unix, unit="s")

        selected_df = budget_df[budget_df["DateTime"] > start_time_datetime]
        col_list = ["Expected", "Expected_max", "Expected_min"]
        for col in col_list:
            selected_df[col] = (budget_df[col] / original_price) * changed_price
        budget_df.update(selected_df)
        monthly_data_mask = budget_df["QHY"].str.startswith("M")
        budget_month = budget_df[monthly_data_mask]
        year_values = budget_month["Year"].unique()
        modified_df = pd.DataFrame()
        year_list = []
        for years in year_values:
            year_df = budget_month[budget_month["Year"] == years]
            year_list.append(years)
            for i in range(4):
                month_list = [f"M{(i*3)+1}", f"M{(i*3)+2}", f"M{(i*3)+3}"]
                filtered_df = year_df[year_df["QHY"].isin(month_list)]
                column_sums = filtered_df[
                    ["Expected", "Expected_min", "Expected_max"]
                ].sum(axis=0)
                filtered_df = pd.DataFrame(column_sums).transpose()
                filtered_df["QHY"] = f"Q{i+1}"
                filtered_df["Year"] = years
                modified_df = pd.concat(
                    [modified_df, filtered_df], axis=0, ignore_index=True
                )
            for i in range(2):
                quarter_list = [f"Q{(i*2)+1}", f"Q{(i*2)+2}"]
                filtered_df = modified_df[modified_df["QHY"].isin(quarter_list)]
                column_sums = filtered_df[
                    ["Expected", "Expected_min", "Expected_max"]
                ].sum(axis=0)
                filtered_df = pd.DataFrame(column_sums).transpose()
                filtered_df["QHY"] = f"H{i+1}"
                filtered_df["Year"] = years
                modified_df = pd.concat(
                    [modified_df, filtered_df], axis=0, ignore_index=True
                )
            column_sums = year_df[["Expected", "Expected_min", "Expected_max"]].sum(
                axis=0
            )
            filtered_df = pd.DataFrame(column_sums).transpose()
            filtered_df["QHY"] = "YY"
            filtered_df["Year"] = years
            modified_df = pd.concat(
                [modified_df, filtered_df], axis=0, ignore_index=True
            )

        self.logger.debug("--------first modified -budget_df------------")
        self.logger.debug(budget_df)
        budget_df1 = pd.merge(budget_df, modified_df, on=["QHY", "Year"], how="left")
        budget_df1.loc[budget_df1["Expected_y"].notnull(), "Expected_x"] = budget_df1[
            "Expected_y"
        ]
        budget_df1.loc[budget_df1["Expected_max_y"].notnull(), "Expected_max_x"] = (
            budget_df1["Expected_max_y"]
        )
        budget_df1.loc[budget_df1["Expected_min_y"].notnull(), "Expected_min_x"] = (
            budget_df1["Expected_min_y"]
        )
        budget_df1 = budget_df1[
            [
                "DateTime",
                "Year",
                "QHY",
                "Expected_x",
                "Expected_min_x",
                "Expected_max_x",
            ]
        ]
        budget_df1 = budget_df1.rename(
            columns={
                "Expected_x": "Expected",
                "Expected_min_x": "Expected_min",
                "Expected_max_x": "Expected_max",
            }
        )
        budget_df = budget_df1
        self.logger.debug("--------final modified -budget_df------------")
        self.logger.debug(budget_df)
        return budget_df
