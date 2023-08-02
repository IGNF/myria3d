# main

# 3.4.3
- Remove outdated and incorrect hydra parameter in config.yaml

# 3.4.2
- Reconstruct absolute path of input LAS files explicitely, removing a costly glob operation.

# 3.4.1
- Fix dataset description for pacasam: there was an unwanted int-to-int mapping in classification_dict.

# 3.4.0
- Allow inference for the smallest possible patches (num_nodes=1) to have consistent inference behavior 