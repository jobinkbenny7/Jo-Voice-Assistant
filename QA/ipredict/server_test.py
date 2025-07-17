from concurrent import futures
import pandas as pd
from datetime import datetime
import grpc
import sys

import ipredict_test_pb2
import ipredict_test_pb2_grpc
from ipredict.algorithm_impl.utils import get_logger
from testing.hyper_parameter_tuning import HyperParameterTuning 
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

class TestingServicer(ipredict_test_pb2_grpc.CostSenseIPredictTestingServicer):
    def __init__(self):
        self.logger = get_logger()
        self.object_HyperParameterTuning = HyperParameterTuning()

    def Hyperparametertuning(self, request, context):
        try:
            folder_path = request.subject
            response = self.object_HyperParameterTuning.find_accuracy(folder_path)
            TestResponse = ipredict_test_pb2.TestResponse()
            TestResponse.testStatus = response
            return TestResponse
        except:
            self.logger.exception("Exception in server Testing........")
    
    def Readingtablenames(self, request, context):
        self.logger.debug(f"reading tables...requests-->{request}")
        sys.stdout.flush()
        try:
             address = request.address
             port_number = request.port
             table_name = request.tablename
             table_response = self.object_HyperParameterTuning.readingtables1(address, port_number, table_name)
             TableResponse = ipredict_test_pb2.TableResponse()
             if len(table_response) != 0:
                for table in table_response:
                    TableResponse.tables.append(table)
             return TableResponse
             
        except:
            self.logger.exception("Exception")

    def trainingvalidation(self, request, context):
        self.logger.debug(f"inside training validation.....")
        sys.stdout.flush()
        try:
            table = request.subject
            growth= request.growth
            growth_expected = ["linear", "logistic"]
            seasonality_mode_expected = ["additive", "multiplicative"]
            # daily_seasonality_expected = ["auto", True, False]
            if request.HasField("growth") and (request.growth in growth_expected):
                growth = request.growth
            else:
                growth = "linear"

            if request.HasField("seasonalityMode") and (request.seasonalityMode in seasonality_mode_expected):
                seasonality_mode = request.seasonalityMode
            else:
                seasonality_mode = "additive"

            if request.HasField("changepointPriorScale"):
                changepoint_prior_scale = request.changepointPriorScale
            else:
                changepoint_prior_scale = 0.05

            if request.HasField("holidaysPriorScale"):
                holidays_prior_scale = request.holidaysPriorScale
            else:
                holidays_prior_scale = 10.0

            if request.HasField("intervalWidth"):
                interval_width = request.intervalWidth
            else:
                interval_width = 0.80

            if request.HasField("seasonalityPriorScale"):
                seasonality_prior_scale = request.seasonalityPriorScale
            else:
                seasonality_prior_scale = 10.0

            if request.HasField("dailySeasonality"):
                daily_seasonality = request.dailySeasonality
            else:
                daily_seasonality = "auto"

            if request.HasField("mcmcSamples"):
                mcmc_samples = request.mcmcSamples
            else:
                mcmc_samples = 0

            response = self.object_HyperParameterTuning.accuracy_model(table, growth, 
                seasonality_mode,
                changepoint_prior_scale, 
                holidays_prior_scale,
                interval_width, 
                seasonality_prior_scale,
                daily_seasonality, 
                mcmc_samples)
            accuracyresponse = ipredict_test_pb2.accuracyResponse()
            accuracyresponse.trainMAE = response["train_mae"]
            accuracyresponse.trainMSE = response["train_mse"]
            accuracyresponse.trainAccuracy = response["train_accuracy"]
            accuracyresponse.testMAE = response["test_mae"]
            accuracyresponse.testMSE = response["test_mse"]
            accuracyresponse.testAccuracy = response["test_accuracy"]
            return accuracyresponse
        except:
            self.logger.exception("Exception in server Testing........")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ipredict_test_pb2_grpc.add_CostSenseIPredictTestingServicer_to_server(TestingServicer(), server)
    server.add_insecure_port("0.0.0.0:50053")
    server.start()
    logger=get_logger()
    logger.debug("testing server started")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()