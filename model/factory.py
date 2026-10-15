"""Model-family dispatch used by the existing train/eval/predict entry points."""
NATIVE_KEYS = ("vlm_model", "vlm_revision", "query_revision", "text_model", "visual_tokens",
               "total_sequence_tokens", "query_tokens", "cost_dim", "quantize_4bit", "lora_rank",
               "gradient_checkpointing", "adapter_control", "plan_control", "profile_stages")


def native_config(args):
    config = {name: getattr(args, name) for name in NATIVE_KEYS}
    for name, default in (("ot_epsilon", .05), ("ot_iterations", 20), ("ot_dustbin_mass", .2),
                          ("ot_dustbin_cost", 1.), ("max_answer_tokens", 64),
                          ("figure_context", True), ("prompt_version", "scivqa-v2")):
        value = getattr(args, name)
        config[name] = default if value is None else value
    config.update(model_family="qwen3_vl", fusion="ot", decode_version="native-greedy-v1")
    return config


def build_native(args, device, resume=None):
    from model.pretrained_vlm import build_pretrained_vlm
    config = resume["model_config"] if resume else native_config(args)
    return build_pretrained_vlm(config, device)
