import yaml
from biotorch.benchmark.run import Benchmark

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Open an example configuration file
config_path = "dfa.yaml"
with open(config_path, 'r') as stream:
    try:
        file = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Deterministic won't work from Colab because we have to set CUDA environments
file['experiment']['deterministic'] = False

with open(config_path, 'w') as outfile:
    yaml.dump(file, outfile)

benchmark = Benchmark(config_path)

benchmark.run()
