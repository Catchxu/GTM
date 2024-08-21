from typing import Dict, Any


def update_configs_with_args(configs, args_dict: Dict[str, Any], suffix):
    for key, value in args_dict.items():
        if suffix is not None:
            if key.endswith(suffix):
                # Remove the suffix
                config_key = key[:-len(suffix)]
                # Only update if the argument is provided and valid
                if hasattr(configs, config_key) and value is not None:
                    setattr(configs, config_key, value)