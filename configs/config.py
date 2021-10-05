import json
import os

from addict import Addict

from utils.addict import unchecked_merge_a_into_b

AVAILABLE_MODELS = ["trajectory2vec", "t2vec"]
AVAILABLE_EXPERIMENTS = ["trajectory-similarity", "trajectory-clustering", "travel-time-prediction"]

configs_dir = os.path.dirname(os.path.abspath(__file__))


def get_experiment_config(model: str, experiment: str):
    if model not in AVAILABLE_MODELS:
        print(f"model config not found: {model}")
        return {}
    model_defaults = Addict(json.load(open(f"{configs_dir}/../resources/config/{model}.default.json")))

    if experiment not in AVAILABLE_EXPERIMENTS:
        print(f"Searching for custom experiment configs: {AVAILABLE_EXPERIMENTS}")
        print(f"Custom experiment parameters for '{experiment}' not found for {model}")
        return model_defaults
    custom_exp_config = Addict(json.load(open(f"{configs_dir}/../resources/config/{model}.{experiment}.json")))

    return unchecked_merge_a_into_b(custom_exp_config, model_defaults)
