# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring,line-too-long,no-member,unused-argument

import logging
import sys

import pandas as pd
from ipredict.gen_code import (ipredict_interface_v1_pb2,
                               ipredict_interface_v1_pb2_grpc)
from ipredict.util.forecast_anomaly_trend import ForecastAnomalyTrend

object_forecast_anomaly_trend = ForecastAnomalyTrend()


class AnalyserServicer(ipredict_interface_v1_pb2_grpc.CostSenseIPredictServicer):
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def Trend(self, request, context):
        self.logger.info("request received from the Trend API -> \n %s", request)
        table = request.subject
        seasonality = request.seasonality
        self.logger.debug(
            "table name %s and seasonality -> %s for trend", table, seasonality
        )
        sys.stdout.flush()
        timezone_str = request.timezone if request.HasField("timezone") else "+00:00"
        if request.HasField("startTimestamp") and (request.startTimestamp != 0):
            start_time = pd.to_datetime(request.startTimestamp, unit="s")
        else:
            start_time = None
        if request.HasField("endTimestamp") and (request.endTimestamp != 0):
            end_time = pd.to_datetime(request.endTimestamp, unit="s")
        else:
            end_time = None
        unit_expected = ["Input", "Month", "Week", "Day", "Hour", "Minute", "Second"]
        if request.HasField("unit") and (request.unit in unit_expected):
            trend_time_type = request.unit
        else:
            trend_time_type = "Input"
        if request.HasField("window") and (request.window != 0):
            trend_time_interval = request.window
        else:
            trend_time_interval = 1
        if seasonality in ("overall", ""):
            seasonality = "trend"
        self.logger.debug(
            "start_time -> %s, end_time -> %s, trend_time_interval -> %s, trend_time_type -> %s ",
            start_time,
            end_time,
            trend_time_interval,
            trend_time_type,
        )
        sys.stdout.flush()
        data_df = object_forecast_anomaly_trend.fetch_trend(
            table,
            seasonality,
            trend_time_type,
            trend_time_interval,
            start_time,
            end_time,
            timezone_str,
        )
        self.logger.info("data_df shape - > %s", data_df.shape)
        final_data = ipredict_interface_v1_pb2.TrendRes()
        return self.generate_trend_response(data_df, final_data)

    def Anomaly(self, request, context):
        self.logger.info("request received from the  Anomaly API -> \n %s", request)
        folder_path = request.subject
        start_time = request.startTimestamp
        end_time = request.endTimestamp
        sys.stdout.flush()
        timezone_str = request.timezone if request.HasField("timezone") else "+00:00"
        if request.startTimestamp != 0:
            start_time = pd.to_datetime(request.startTimestamp, unit="s")
        else:
            start_time = None
        if request.endTimestamp != 0:
            end_time = pd.to_datetime(request.endTimestamp, unit="s")
        else:
            end_time = None
        filter_expected = ["All", "Anomaly", "Normal"]
        if request.HasField("filter") and (request.filter in filter_expected):
            anomaly_filter = request.filter
        else:
            anomaly_filter = "All"
        unit_expected = ["Input", "Month", "Week", "Day", "Hour", "Minute", "Second"]
        if request.HasField("unit") and (request.unit in unit_expected):
            anomaly_time_type = request.unit
        else:
            anomaly_time_type = "Input"
        if request.HasField("window") and (request.window != 0):
            anomaly_time_interval = request.window
        else:
            anomaly_time_interval = 1
        self.logger.debug(
            "anomaly_filter - > %s ,  anomaly_time_interval - > %s , anomaly_time_type - > %s ",
            anomaly_filter,
            anomaly_time_interval,
            anomaly_time_type,
        )
        sys.stdout.flush()
        data_df = object_forecast_anomaly_trend.fetch_anomaly(
            folder_path,
            anomaly_filter,
            anomaly_time_type,
            anomaly_time_interval,
            start_time,
            end_time,
            timezone_str,
        )
        self.logger.info("anomaly df shape - > %s", data_df.shape)
        final_data = ipredict_interface_v1_pb2.AnomalyRes()
        query_response = ipredict_interface_v1_pb2.Anomaly()
        return self.generate_anomaly_response(data_df, final_data, query_response)

    def AnomalyReport(self, request, context):
        self.logger.info(
            "request received from the AnomalyReport API -> \n %s", request
        )
        folder_path = request.subject
        start_time = request.startTimestamp
        end_time = request.endTimestamp
        self.logger.debug(
            "folder_path - > %s ,  start_time - > %s , end_time - > %s ",
            folder_path,
            start_time,
            end_time,
        )
        sys.stdout.flush()
        if request.startTimestamp != 0:
            start_time = pd.to_datetime(request.startTimestamp, unit="s")
        else:
            start_time = None
        if request.endTimestamp != 0:
            end_time = pd.to_datetime(request.endTimestamp, unit="s")
        else:
            end_time = None
        unit_expected = ["Input", "Month", "Week", "Day", "Hour", "Minute", "Second"]
        if request.HasField("unit") and (request.unit in unit_expected):
            anomaly_time_type = request.unit
        else:
            anomaly_time_type = "Hour"
        if request.HasField("window") and (request.window != 0):
            anomaly_time_interval = request.window
        else:
            anomaly_time_interval = 1
        self.logger.debug(
            "anomaly_time_interval - > %s , anomaly_time_type - > %s ",
            anomaly_time_interval,
            anomaly_time_type,
        )
        sys.stdout.flush()
        response = object_forecast_anomaly_trend.fetch_anomaly_report(
            folder_path, anomaly_time_type, anomaly_time_interval, start_time, end_time
        )
        report_response = ipredict_interface_v1_pb2.AnomalyReportRes()
        if response is None or (isinstance(response, pd.DataFrame) and response.empty):
            return report_response
        report_item = ipredict_interface_v1_pb2.AnomalyReport(
            max_timestamp=response["max_anomaly_date"],
            num_anomalies=response["num_anomalies"],
            Anomaly_percentage=response["Anomaly_percentage"],
        )

        report_response.data.append(report_item)
        return report_response

    def Budget(self, request, context):
        self.logger.info("request received from the Budget API -> \n %s", request)
        folder_path = request.subject
        self.logger.debug(folder_path)
        data_df = object_forecast_anomaly_trend.get_budget(table=folder_path)
        self.logger.info(data_df.shape)
        final_data = ipredict_interface_v1_pb2.BudgetRes()
        query_response = ipredict_interface_v1_pb2.Budget()
        return self.generate_budget_response(data_df, final_data, query_response)

    def NewBudget(self, request, context):
        self.logger.info("request received from the New Budget API -> \n %s", request)
        parent_table = request.parentsubject
        self.logger.debug(parent_table)
        effecting_table = request.effectingsubject
        self.logger.debug(effecting_table)
        data_df = object_forecast_anomaly_trend.get_new_budget(
            parent_table=parent_table, effecting_table=effecting_table
        )
        self.logger.debug(data_df.shape)
        final_data = ipredict_interface_v1_pb2.NewBudgetRes()
        query_response = ipredict_interface_v1_pb2.NewBudget()
        return self.generate_budget_response(data_df, final_data, query_response)

    def NewPrice(self, request, context):
        self.logger.info(
            "request received from the Price change API -> \n %s", request
        )
        table = request.parentsubject
        self.logger.debug(table)
        original_price = request.value
        self.logger.debug(original_price)
        month = request.month
        self.logger.debug(month)
        year = request.year
        self.logger.debug(year)
        changed_price = request.changedValue
        self.logger.debug(changed_price)
        data_df = object_forecast_anomaly_trend.get_price_change(
            table=table,
            month=month,
            year=year,
            original_price=original_price,
            changed_price=changed_price,
        )
        self.logger.debug(data_df.shape)
        final_data = ipredict_interface_v1_pb2.NewPriceRes()
        query_response = ipredict_interface_v1_pb2.NewPrice()
        return self.generate_budget_response(data_df, final_data, query_response)

    def generate_forecasting_response(self, data_df, final_data, query_response):
        if data_df.shape[0] != 0:
            for _, row in data_df.iterrows():
                query_response.startTime = row["start_time"]
                query_response.endTime = row["end_time"]
                query_response.expected = max(0.0, row["Expected"])
                query_response.expectedMax = max(0.0, row["Expected_max"])
                query_response.expectedMin = max(0.0, row["Expected_min"])
                final_data.data.append(query_response)
        return final_data

    def generate_budget_response(self, data_df, final_data, query_response):
        if data_df.shape[0] != 0:
            for _, row in data_df.iterrows():
                query_response.dateTime = row["DateTime"]
                query_response.expected = max(0.0, row["Expected"])
                query_response.expectedMax = max(0.0, row["Expected_max"])
                query_response.expectedMin = max(0.0, row["Expected_min"])
                query_response.qhy = row["QHY"]
                query_response.year = row["Year"]
                final_data.data.append(query_response)
        return final_data

    def generate_anomaly_response(self, data_df, final_data, query_response):
        if data_df.shape[0] != 0:
            for _, row in data_df.iterrows():
                query_response.startTime = row["start_time"]
                query_response.endTime = row["end_time"]
                query_response.expected = row["Expected"]
                query_response.expectedMax = row["Expected Maximum"]
                query_response.expectedMin = row["Expected Minimum"]
                query_response.currentValue = row["Current Value"]
                query_response.deviation = row["Deviation"]
                query_response.isAnomaly = not row["Anomaly"] == "Normal"
                final_data.data.append(query_response)
        return final_data

    def generate_trend_response(self, data_df, final_data):
        query_response = ipredict_interface_v1_pb2.TrendData()
        if data_df.shape[0] != 0:
            for _, row in data_df.iterrows():
                query_response.startTime = data_df.at[row.name, "start_time"]
                query_response.endTime = data_df.at[row.name, "end_time"]
                query_response.trendValue = row["trend"]
                query_response.trendLower = row["trend_lower"]
                query_response.trendUpper = row["trend_upper"]
                query_response.currentValue = row["Current Value"]
                final_data.data.append(query_response)
        return final_data
