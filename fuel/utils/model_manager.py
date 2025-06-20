from loguru import logger

from ..model import AlsServerModel, GenLocalModel, GenServerModel
from ..guidance.Random import Random
from ..guidance.SA import SA
from .util import load_config_file


class ModelManager:
    """Utility class for managing models and heuristic algorithms"""
    
    @staticmethod
    def setup_models(model_config, use_local_gen=False):
        """Setup generation model and analysis model"""
        if use_local_gen:
            gen_model = GenLocalModel(config=model_config)
            logger.info(f"[USE_LOCAL_GEN] GEN MODEL: {model_config['local']['model']}")
        else:
            gen_model = GenServerModel(config=model_config)
            logger.info(f"[USE_SERVER_GEN] GEN MODEL: {model_config['server']['model']}")
        
        als_model = AlsServerModel(config=model_config)
        return gen_model, als_model
    
    @staticmethod
    def setup_heuristic(heuristic_name, heuristic_config_path, op_set):
        """Setup heuristic algorithm"""
        heuristic_name = heuristic_name.replace("\r", "")
        
        if heuristic_name == "None":
            logger.info("[HEURISTIC]: None!")
            return None
        
        heuristic_config = load_config_file(heuristic_config_path)[heuristic_name]
        
        if heuristic_name == "SA":
            logger.info("[HEURISTIC]: Simulated Annealing!")
            return SA(
                float(heuristic_config["t0"]),
                float(heuristic_config["tmin"]),
                float(heuristic_config["alpha"]),
                op_set,
            )
        elif heuristic_name == "Random":
            logger.info("[HEURISTIC]: Random!")
            return Random(
                int(heuristic_config["l"]), 
                int(heuristic_config["r"]), 
                op_set
            )
        
        return None 