# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring,wrong-import-position,too-many-positional-arguments,too-many-arguments

import logging
import os
import sys
from concurrent import futures
from datetime import datetime

import fire
import grpc

sys.path.append(f"{os.path.dirname(__file__)}/../..")
from ipredict.algorithm_impl.utils import init_logger
from ipredict.config import config
from ipredict.config.config import init_config
from ipredict.data_processor.forecast_processor import Predict, forecastService
from ipredict.gen_code import ipredict_forecast_pb2_grpc
from ipredict.util.forecast_anomaly_trend import ForecastAnomalyTrend

TRAIN_TIME_TYPE = "Input"
TRAIN_TIME_INTERVAL = 1
MAX_YEAR_BUDGET_COUNT = 10
MIN_YEAR_BUDGET_COUNT = 2
MONTH = "01"
START_YEAR = 2024
YEAR_COUNT = 1
ENV_PATH = "./config.env"
PORT = "50052"


class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.object_forecast_anomaly_trend = ForecastAnomalyTrend()
        self.object_predict = None

    def training(
        self,
        subject,
        env_path=ENV_PATH,
        train_time_type=TRAIN_TIME_TYPE,
        train_time_interval=TRAIN_TIME_INTERVAL,
    ):
        init_config(env_path)

        try:
            self.logger.debug("Training started ........")
            unit_expected = [
                "Input",
                "Month",
                "Week",
                "Day",
                "Hour",
                "Minute",
                "Second",
            ]
            train_time_type = (
                train_time_type if train_time_type in unit_expected else "Input"
            )
            train_time_interval = train_time_interval if train_time_interval > 0 else 1

            self.logger.debug("Train time type -> %s", train_time_type)
            self.logger.debug("Train time interval -> %s", train_time_interval)

            self.object_forecast_anomaly_trend.train_model_calculate_trend_anomaly(
                subject, train_time_type, train_time_interval
            )

        except ValueError:
            self.logger.error("ValueError: training failed")
            sys.exit(int(config.VALUE_ERROR_CODE))

    def budget(
        self, subject, env_path=ENV_PATH, month="01", start_year=2024, year_count=2
    ):
        init_config(env_path)
        try:
            month = MONTH
            start_year = START_YEAR
            year_count = YEAR_COUNT

            month_expected = [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ]
            if month not in month_expected:
                month = "01"

            current_year = datetime.today().year
            if start_year < (current_year - 1):
                pass
            else:
                start_year = current_year

            max_budget_year_count = MAX_YEAR_BUDGET_COUNT
            if year_count > 1:
                year_count = min(year_count, max_budget_year_count)
            else:
                min_budget_year_count = MIN_YEAR_BUDGET_COUNT
                year_count = min_budget_year_count
            self.logger.info("Budget calculation starts from month: %s and year: %s",
                              month, start_year)

            sys.stdout.flush()

            self.object_forecast_anomaly_trend.calculate_budget(
                subject, start_year, month, year_count
            )

        except ValueError:
            self.logger.error("ValueError: BudgetCalculation failed")
            sys.exit(int(config.VALUE_ERROR_CODE))

    def forecast(self, folder_path, period, unit, window, env_path=ENV_PATH):
        init_config(env_path)
        self.object_predict = Predict()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        ipredict_forecast_pb2_grpc.add_forecastServicer_to_server(
            forecastService(), server
        )
        port = os.getenv("PORT", PORT)
        server_address = f"0.0.0.0:{port}"
        self.logger.debug("Starting server at %s", server_address)
        server.add_insecure_port(server_address)
        server.start()
        self.logger.info("request received from the Forecasting JOB -> ")
        self.logger.info("Forecast for %s", folder_path)
        self.object_predict.forecasting(
            period=period, unit=unit, window=window, folder_path=folder_path
        )
        server.wait_for_termination()


if __name__ == "__main__":
    init_logger()
    try:
        fire.Fire(DataProcessor())
    except Exception as e:
        logging.exception(
            "An unexpected error occurred during command execution.", exc_info=True
        )
        sys.exit(int(config.VALUE_ERROR_CODE))
