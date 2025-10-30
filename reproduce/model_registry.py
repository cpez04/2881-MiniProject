"""
Model registry for multi-model inference.
Maps friendly names to provider-specific model IDs and configurations.
"""

MODEL_REGISTRY = {
    "llama2-7b-chat": {
        "local_id": "meta-llama/Llama-2-7b-chat-hf",
        "together_id": "meta-llama/Llama-2-7b-chat-hf",
        "context_window": 4096,
        "size": "7b",
        "license": "llama2",
        "recommended_backend": "local",
    },
    "mistral-7b-instruct": {
        "local_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "together_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "context_window": 8192,
        "size": "7b",
        "license": "apache-2.0",
        "recommended_backend": "local",
    },
    "solar-10.7b": {
        "local_id": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "together_id": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "context_window": 4096,
        "size": "13b",
        "license": "apache-2.0",
        "recommended_backend": "api",
    },
    "llama2-13b-chat": {
        "local_id": "meta-llama/Llama-2-13b-chat-hf",
        "together_id": "meta-llama/Llama-2-13b-chat-hf",
        "context_window": 4096,
        "size": "13b",
        "license": "llama2",
        "recommended_backend": "api",
    },
    "vicuna-13b": {
        "local_id": "lmsys/vicuna-13b-v1.5",
        "together_id": "lmsys/vicuna-13b-v1.5",
        "context_window": 4096,
        "size": "13b",
        "license": "llama2",
        "recommended_backend": "api",
    },
    "mixtral-8x7b-instruct": {
        "local_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "together_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "context_window": 32768,
        "size": "8x7b",
        "license": "apache-2.0",
        "recommended_backend": "api",
    },
    "wizardlm-13b": {
        "local_id": "WizardLM/WizardLM-13B-V1.2",
        "together_id": "WizardLM/WizardLM-13B-V1.2",
        "context_window": 4096,
        "size": "13b",
        "license": "llama2",
        "recommended_backend": "api",
    },
}

def get_model_info(model_key):
    """Get model information from registry."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_key} not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_key]

def get_model_id(model_key, backend="local"):
    """Get the appropriate model ID for the specified backend."""
    info = get_model_info(model_key)
    if backend == "local":
        return info["local_id"]
    elif backend in ["together", "api"]:
        return info["together_id"]
    else:
        raise ValueError(f"Unknown backend: {backend}")

def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())
