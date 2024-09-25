import yaml
from federated_learning import FederatedLearning
from memory_network import MemoryNetwork
from adaptive_model import AdaptiveModel

# Load experiment config
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_file):
    config = load_config(config_file)
    # Example: Setup clients, server, model, etc.
    # federated = FederatedLearning(clients, server, model, config['federated_rounds'])
    # federated.train()
    print("Training process started...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train FedCL-LLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    main(args.config)
