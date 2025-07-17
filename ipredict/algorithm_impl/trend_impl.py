# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring,too-many-positional-arguments,too-many-arguments,too-many-locals,too-many-branches

import logging
from datetime import timedelta
import numpy as np
import pandas as pd

import ipredict.config.config
from ipredict.algorithm_impl.utils import (
    FREQ_DICT, execute_query_get_df,
    get_questdb_data_query_with_start_end_date_mandatory,
    get_resample_start_end_date_from_input, read_data_df, resample_data,
    save_data_df)


class TrendImpl:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def seasonality_plot_df_impl(self, m, ds):
        self.logger.debug("function -> seasonality_plot_df_impl")
        df_dict = {"ds": ds, "cap": 1.0, "floor": 0.0}
        for name in m.extra_regressors:
            df_dict[name] = 0.0
        for props in m.seasonalities.values():
            if props["condition_name"] is not None:
                df_dict[props["condition_name"]] = True
        df = pd.DataFrame(df_dict)
        df = m.setup_dataframe(df)
        return df

    def plot_weekly_impl(self, m, table, weekly_start=0, name="weekly"):
        self.logger.debug("function -> plot_weekly_impl")
        days = pd.date_range(start="2017-01-01", periods=7) + pd.Timedelta(
            days=weekly_start
        )
        df_w = self.seasonality_plot_df_impl(m, days)
        seas = m.predict_seasonal_components(df_w)
        seas["ds"] = df_w["ds"].dt.to_pydatetime()
        df = seas[["ds", name, name + "_lower", name + "_upper"]]
        table_name = f"{table}_weekly"
        df = df.reset_index().rename(
            columns={
                name: "trend",
                name + "_lower": "trend_lower",
                name + "_upper": "trend_upper",
            }
        )
        df = df[["ds", "trend", "trend_lower", "trend_upper"]]
        save_data_df("trend_data", table_name, df, write_mode="replace")

    def plot_yearly_impl(self, m, table, yearly_start=0, name="yearly"):
        self.logger.debug("function -> plot_yearly_impl")
        days = pd.date_range(start="2017-01-01", periods=365) + pd.Timedelta(
            days=yearly_start
        )
        df_y = self.seasonality_plot_df_impl(m, days)
        seas = m.predict_seasonal_components(df_y)
        seas["ds"] = df_y["ds"].dt.to_pydatetime()
        df = seas[["ds", name, name + "_lower", name + "_upper"]]
        table_name = f"{table}_yearly"
        df = df.reset_index().rename(
            columns={
                name: "trend",
                name + "_lower": "trend_lower",
                name + "_upper": "trend_upper",
            }
        )
        df = df[["ds", "trend", "trend_lower", "trend_upper"]]
        save_data_df("trend_data", table_name, df, write_mode="replace")

    def plot_seasonality_impl(self, m, name, table):
        self.logger.debug(
            "function -> plot_seasonality_impl -> with plotname -> %s", name
        )
        start = pd.to_datetime("2017-01-01 0000")
        period = m.seasonalities[name]["period"]
        if name == "daily":
            plot_points = 24
            days = [start + pd.Timedelta(hours=i) for i in range(plot_points)]
        elif name == "weekly":
            plot_points = 7
            days = [start + pd.Timedelta(days=i) for i in range(plot_points)]
        else:
            end = start + pd.Timedelta(days=period)
            plot_points = ipredict.config.config.PLOT_POINTS
            days = pd.to_datetime(np.linspace(start.value, end.value, plot_points))
        df_y = self.seasonality_plot_df_impl(m, days)
        seas = m.predict_seasonal_components(df_y)
        seas["ds"] = df_y["ds"].dt.to_pydatetime()
        df = seas[["ds", name, name + "_lower", name + "_upper"]]
        table_name = f"{table}_{name}"
        df = df.reset_index().rename(
            columns={
                name: "trend",
                name + "_lower": "trend_lower",
                name + "_upper": "trend_upper",
            }
        )
        df["Current Value"] = df["trend"]
        df = df[["ds", "trend", "trend_lower", "trend_upper","Current Value"]]
        save_data_df("trend_data", table_name, df, write_mode="replace")

    def plot_forecast_component_impl(self, fcst, name, table, min_val, lower_val):
        self.logger.debug("function -> plot_forecast_component_impl")
        table_name = f"{table}_{name}"
        name = "yhat"
        fcst["ds"] = fcst["ds"].dt.to_pydatetime()
        df = fcst[["ds", name, name + "_lower", name + "_upper","fact"]]
        if min_val is True:
            try:
                df.loc[:, name] = df[name].clip(lower=lower_val)
                df.loc[:, name + "_lower"] = df[name + "_lower"].clip(lower=lower_val)
                df.loc[:, name + "_upper"] = df[name + "_upper"].clip(lower=lower_val)
            except Exception:
                df.loc[name] = df[name].clip(lower=lower_val)
                df.loc[name + "_lower"] = df[name + "_lower"].clip(lower=lower_val)
                df.loc[name + "_upper"] = df[name + "_upper"].clip(lower=lower_val)

        df = df.reset_index().rename(
            columns={
                name: "trend",
                name + "_lower": "trend_lower",
                name + "_upper": "trend_upper",
                         "fact": "Current Value"
            }
        )
        df = df[["ds", "trend", "trend_lower", "trend_upper","Current Value"]]
        save_data_df("trend_data", table_name, df, write_mode="replace")

    def plot_components_impl(
        self, m, fcst, table, min_val, lower_val, weekly_start=0, yearly_start=0
    ):
        self.logger.debug("function -> plot_components_impl")
        components = ["trend"]
        if m.train_holiday_names is not None and "holidays" in fcst:
            components.append("holidays")
        if "weekly" in m.seasonalities and "weekly" in fcst:
            components.append("weekly")
        if "yearly" in m.seasonalities and "yearly" in fcst:
            components.append("yearly")
        components.extend(
            [
                name
                for name in sorted(m.seasonalities)
                if name in fcst and name not in ["weekly", "yearly"]
            ]
        )
        regressors = {"additive": False, "multiplicative": False}
        for name, props in m.extra_regressors.items():
            regressors[props["mode"]] = True
        for mode in ["additive", "multiplicative"]:
            if regressors[mode] and "extra_regressors_{mode}" in fcst:
                components.append("extra_regressors_{mode}")
        multiplicative_axes = []
        dt = m.history["ds"].diff()
        min_dt = dt.iloc[dt.values.nonzero()[0]].min()
        for plot_name in components:
            if plot_name == "trend":
                self.plot_forecast_component_impl(
                    fcst=fcst,
                    name="trend",
                    table=table,
                    min_val=min_val,
                    lower_val=lower_val,
                )

            elif plot_name in m.seasonalities:
                if (
                    plot_name == "weekly" or m.seasonalities[plot_name]["period"] == 7
                ) and (min_dt == pd.Timedelta(days=1)):
                    self.plot_weekly_impl(m=m, table=table, weekly_start=weekly_start)
                elif (
                    plot_name == "yearly"
                    or m.seasonalities[plot_name]["period"] == 365.25
                ):
                    self.plot_yearly_impl(m=m, table=table, yearly_start=yearly_start)
                else:
                    self.plot_seasonality_impl(m=m, name=plot_name, table=table)
            elif plot_name in [
                "holidays",
                "extra_regressors_additive",
                "extra_regressors_multiplicative",
            ]:
                self.plot_forecast_component_impl(
                    fcst=fcst,
                    name=plot_name,
                    table=table,
                    min_val=min_val,
                    lower_val=lower_val,
                )
            if plot_name in m.component_modes["multiplicative"]:
                multiplicative_axes.append(plot_name)
        data = [
            ["components", f"{components}"],
            ["multiplicative_components", f"{multiplicative_axes}"],
        ]
        df = pd.DataFrame(data, columns=["parameter", "values"])
        save_data_df("trend_param", table, df, write_mode="replace")

    def data_trend(self, table, model, analysis_fcst, min_val, lower_val):
        self.logger.debug("function -> get_trend")
        self.plot_components_impl(
            model, analysis_fcst, table, min_val=min_val, lower_val=lower_val
        )

    def trend_agg(self, data_df, trend_time_type, trend_time_interval):
        start_time, end_time = get_resample_start_end_date_from_input(
            data_df, trend_time_type, trend_time_interval
        )
        if (start_time is None) and (end_time is None):
            return data_df
        resample_df = data_df[["ds", "trend", "trend_lower", "trend_upper","Current Value"]]
        resample_df = resample_df[resample_df["ds"] >= start_time]
        resample_df = resample_df[resample_df["ds"] < end_time]
        resampled_df = resample_data(
            resample_df, trend_time_type, "sum", trend_time_interval, label="right"
        )
        return resampled_df

    def read_trend_data_df(
        self,
        database,
        table,
        trend_time_type,
        trend_time_interval,
        start_time,
        end_time,
        timezone_str
    ):
        freq_dict = FREQ_DICT
        if trend_time_type != "Input":
            trend_time_type = freq_dict[trend_time_type]
        if (start_time is None) or (end_time is None):
            data_df = read_data_df("trend_data", table, start_time, end_time)
        else:
            query = get_questdb_data_query_with_start_end_date_mandatory(
                database, table, start_time, end_time
            )
            data_df = execute_query_get_df(query)
            data_df["ds"] = pd.to_datetime(data_df["ds"])

        if (trend_time_type != "Input") and (data_df.shape[0] != 0):
            label_val = "left"
            start_val = data_df["ds"].iloc[0]
            end_val = data_df["ds"].iloc[-1]
            data_df = resample_data(
                data_df, trend_time_type, "sum", trend_time_interval, label=label_val,
                timezone_str=timezone_str, start_time=start_time, end_time=end_time
            )
            if label_val == "right":
                data_df.loc[data_df.index[-1], "ds"] = end_val
            else:
                data_df.loc[data_df.index[0], "ds"] = start_val

        return data_df

    def get_trend_data(
        self,
        table,
        seasonality,
        trend_time_type,
        trend_time_interval,
        start_time,
        end_time,
        timezone_str
    ):
        table = f"{table}_{seasonality}"
        if seasonality != "trend":
            df = read_data_df("trend_data", table)
            df["start_time"] = df["ds"]
            df["end_time"] = df["ds"].shift(-1)
            if seasonality == "weekly":
                df.loc[df.index[-1], "end_time"] = df["start_time"].iloc[-1] + timedelta(days=1)
            elif seasonality == "daily":
                df.loc[df.index[-1], "end_time"] = df["start_time"].iloc[-1] + timedelta(hours=1)
            return df
        return self.read_trend_data_df(
            "trend_data",
            table,
            trend_time_type,
            trend_time_interval,
            start_time,
            end_time,
            timezone_str
        )
