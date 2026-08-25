import json
from axolotl.utils.schemas.fsdp import FSDPConfig

properties = FSDPConfig.model_json_schema()["properties"]

names = [
    "offload_params",
    "cpu_ram_efficient_loading",
    "auto_wrap_policy",
    "transformer_layer_cls_to_wrap",
    "state_dict_type",
    "reshard_after_forward",
    "use_orig_params",
    "activation_checkpointing",
    "sync_module_states",
    "limit_all_gathers",
]

for name in names:
    print()
    print(name)
    print(json.dumps(properties.get(name, "NOT SUPPORTED"), indent=2))
