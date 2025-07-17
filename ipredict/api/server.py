# pylint: disable=missing-function-docstring,missing-module-docstring,missing-class-docstring,line-too-long,wrong-import-position,too-few-public-methods

import logging
import os
import sys
from concurrent import futures

import fire
import grpc

sys.path.append(f"{os.path.dirname(__file__)}/../..")
from ipredict.algorithm_impl.utils import init_logger
from ipredict.api.analyser_service import AnalyserServicer
from ipredict.api.training_service import TrainingServicer
from ipredict.config.config import init_config
from ipredict.gen_code import ipredict_interface_v1_pb2_grpc


class Server:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def serve(self, env_path="./config.env"):
        init_config(env_path)
        logger = logging.getLogger("main")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        ipredict_interface_v1_pb2_grpc.add_CostSenseIPredictTrainingServicer_to_server(
            TrainingServicer(), server
        )
        ipredict_interface_v1_pb2_grpc.add_CostSenseIPredictServicer_to_server(
            AnalyserServicer(), server
        )
        server.add_insecure_port("0.0.0.0:50051")
        server.start()
        logger.info("starting server")
        server.wait_for_termination()


if __name__ == "__main__":
    init_logger()
    fire.Fire(Server())
