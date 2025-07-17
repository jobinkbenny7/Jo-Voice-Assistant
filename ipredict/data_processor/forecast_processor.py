# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring,wrong-import-position,unused-argument,duplicate-code,no-member,invalid-name

import logging
import multiprocessing
import os
import sys

sys.path.append(f"{os.path.dirname(__file__)}/../..")
from ipredict.algorithm_impl.utils import init_logger
from ipredict.gen_code import ipredict_forecast_pb2, ipredict_forecast_pb2_grpc
from ipredict.util.forecast_anomaly_trend import ForecastAnomalyTrend

ENV_PATH = "./config.env"
PORT = 50052

manager = multiprocessing.Manager()
job_status = manager.dict(status="IN PROGRESS", data=None, message="")

init_logger()
logger = logging.getLogger(__name__)


class Predict:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def forecasting(self, folder_path, period, unit, window):
        job_status["status"] = "IN PROGRESS"
        job_status["message"] = "Job is running"
        job_status["data"] = None

        process = multiprocessing.Process(
            target=self.run_forecast_job, args=(folder_path, period, unit, window)
        )
        process.start()

        self.logger.info("Forecasting job started.")
        process.join()

    def run_forecast_job(self, folder_path, period, unit, window):
        try:
            object_forecast_anomaly_trend = ForecastAnomalyTrend()
            self.logger.info("Running forecasting process.")

            data_df = object_forecast_anomaly_trend.forecasting(
                periods=period,
                frequency_type=unit,
                frequency=window,
                model_folder_path=folder_path,
                min_val=True,
                lower_val=0.0,
            )

            job_status["status"] = "SUCCESS"
            job_status["data"] = data_df.to_dict(orient="records")
            job_status["message"] = "Forecasting completed successfully"

            self.logger.info("Forecasting completed successfully.")
        except Exception:
            self.logger.error("Forecasting failed")
            job_status["status"] = "FAILED"
            job_status["message"] = "Forecasting failed"


class forecastService(
    ipredict_forecast_pb2_grpc.forecastServicer
):  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def GetJobStatus(self, request, context):

        status_data = job_status
        response = ipredict_forecast_pb2.JobStatusUpdate(
            status=status_data["status"], message=status_data["message"]
        )

        if status_data["status"] == "SUCCESS" and status_data["data"]:
            for record in status_data["data"]:
                try:
                    response.data.add(
                        startTime=record["start_time"],
                        endTime=record["end_time"],
                        expected=record["Expected"],
                        expectedMax=record["Expected_max"],
                        expectedMin=record["Expected_min"],
                    )
                except Exception as e:
                    self.logger.error("Error processing record: %s - %s", record, e)

        return response
