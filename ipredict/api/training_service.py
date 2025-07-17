# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring,line-too-long,no-member,unused-argument


import logging
import sys
from datetime import datetime

import pandas as pd
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

from ipredict.gen_code import (ipredict_interface_v1_pb2,
                               ipredict_interface_v1_pb2_grpc)
from ipredict.util.forecast_anomaly_trend import ForecastAnomalyTrend

object_forecast_anomaly_trend = ForecastAnomalyTrend()


class TrainingServicer(
    ipredict_interface_v1_pb2_grpc.CostSenseIPredictTrainingServicer
):
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def CurrentStatus(self, request, context):
        self.logger.info("request received from the CurrentStatus API -> \n %s", request)
        sys.stdout.flush()
        folder_path = request.subject
        response = object_forecast_anomaly_trend.get_current_state(folder_path, context)
        if response is None:
            return ipredict_interface_v1_pb2.StatusResponse()
        status_response = ipredict_interface_v1_pb2.StatusResponse()
        status_response.modelExist = response["model_exist"]
        status_response.modelStartTimestamp = response["model_start_time"]
        status_response.modelEndTimestamp = response["model_end_time"]
        status_response.inputDataStartTimestamp = response["input_data_start_time"]
        status_response.inputDataEndTimestamp = response["input_data_end_time"]
        status_response.budgetCalculated = response["budget_calculated"]
        status_response.modeltimediff = response["input_data_time_diff_in_sec"]
        return status_response

    def PushMetrics(self, request, context):
        self.logger.info("request received from the PushMetrics API ->")
        subject = request.meta.subject
        self.logger.info("subject -> %s", subject)
        folder_path = ""
        time_stamp = []
        label = []
        data_df = pd.DataFrame()
        folder_path = request.meta.subject
        for metric in request.data:
            date_time = datetime.fromtimestamp(metric.timestamp)
            time_stamp.append(date_time)
            label.append(metric.value)
        data_df["ds"] = time_stamp
        data_df["y"] = label
        object_forecast_anomaly_trend.push_input_data(folder_path, data_df)
        return google_dot_protobuf_dot_empty__pb2.Empty()

    def DeleteTable(self, request, context):
        self.logger.info("request received from the DeleteTable API -> \n %s", request)
        subject = request.subject
        object_forecast_anomaly_trend.delete_data(subject)
        return google_dot_protobuf_dot_empty__pb2.Empty()
