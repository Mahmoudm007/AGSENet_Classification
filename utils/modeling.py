import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from models.agsenet_classifier import AGSENetClassifier
from .class_descriptions import get_class_description, get_class_display_name
from .tfidf_text import encode_texts_with_tfidf


def get_description_aux_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(config.get("description_aux", {}))


def description_aux_enabled(config: Dict[str, Any]) -> bool:
    return bool(get_description_aux_config(config).get("enabled", False))


def load_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload
    if isinstance(payload, dict):
        return {"state_dict": payload}
    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def _build_description_prompts(class_names: List[str]) -> List[str]:
    prompts: List[str] = []
    for class_name in class_names:
        display_name = get_class_display_name(class_name)
        description = get_class_description(class_name)
        prompts.append(
            f"Road surface class: {display_name}. "
            f"Detailed condition description: {description}"
        )
    return prompts


def resolve_description_embeddings(
    config: Dict[str, Any],
    class_names: List[str],
    device: torch.device,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    aux_cfg = get_description_aux_config(config)
    if state_dict is not None:
        has_aux_keys = (
            "description_embeddings" in state_dict
            or any(key.startswith("image_projection.") for key in state_dict.keys())
            or any(key.startswith("description_adapter.") for key in state_dict.keys())
        )
        if not has_aux_keys:
            return None, {
                "enabled": False,
                "source": "checkpoint_without_aux",
                "class_names": class_names,
            }

    if state_dict is not None and "description_embeddings" in state_dict:
        embeddings = state_dict["description_embeddings"].detach().cpu().to(torch.float32)
        return embeddings, {
            "enabled": True,
            "source": "checkpoint",
            "text_feature_extractor": aux_cfg.get("text_feature_extractor", "tfidf"),
            "class_names": class_names,
        }

    if not aux_cfg.get("enabled", False):
        return None, {"enabled": False, "source": "disabled", "class_names": class_names}

    prompts = _build_description_prompts(class_names)
    ngram_range = aux_cfg.get("tfidf_ngram_range", [1, 2])
    embeddings, text_meta = encode_texts_with_tfidf(
        prompts,
        max_features=int(aux_cfg.get("tfidf_max_features", 4000)),
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
        use_svd=bool(aux_cfg.get("tfidf_use_svd", False)),
        svd_dim=int(aux_cfg.get("tfidf_svd_dim", 128)),
        seed=int(config.get("seed", 42)),
        top_k_terms=int(aux_cfg.get("tfidf_top_terms", 12)),
        use_char_ngrams=bool(aux_cfg.get("tfidf_use_char_ngrams", True)),
        char_max_features=int(aux_cfg.get("tfidf_char_max_features", 2000)),
        char_ngram_range=(
            int(aux_cfg.get("tfidf_char_ngram_range", [3, 5])[0]),
            int(aux_cfg.get("tfidf_char_ngram_range", [3, 5])[1]),
        ),
        sublinear_tf=bool(aux_cfg.get("tfidf_sublinear_tf", True)),
    )
    metadata = {
        "enabled": True,
        "source": "tfidf_vectorizer",
        "text_feature_extractor": aux_cfg.get("text_feature_extractor", "tfidf"),
        "class_names": class_names,
        "display_names": [get_class_display_name(name) for name in class_names],
        "descriptions": {name: get_class_description(name) for name in class_names},
        "prompts": prompts,
    }
    metadata.update(text_meta)
    return embeddings, metadata


def build_model(
    config: Dict[str, Any],
    class_names: List[str],
    device: torch.device,
    checkpoint_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[AGSENetClassifier, Dict[str, Any]]:
    model_config = checkpoint_payload.get("config", config) if checkpoint_payload else config
    state_dict = checkpoint_payload.get("state_dict") if checkpoint_payload else None
    description_embeddings, text_metadata = resolve_description_embeddings(
        config=model_config,
        class_names=class_names,
        device=device,
        state_dict=state_dict,
    )

    aux_cfg = get_description_aux_config(model_config)
    model = AGSENetClassifier(
        in_ch=3,
        out_ch=len(class_names),
        base_ch=int(model_config["model_channels"]),
        dropout=float(model_config["dropout"]),
        description_embeddings=description_embeddings,
        description_mix_weight=float(aux_cfg.get("mix_weight", 0.35)),
        description_hidden_dim=int(aux_cfg.get("hidden_dim", 512)),
        description_logit_scale_init=float(aux_cfg.get("logit_scale_init", 14.2857)),
        scale_attention_heads=int(aux_cfg.get("scale_attention_heads", 4)),
        fusion_gate_hidden_dim=int(aux_cfg.get("fusion_gate_hidden_dim", 256)),
    ).to(device)
    return model, text_metadata


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_payload: Dict[str, Any],
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    unexpected: List[str] = []
    state_dict = checkpoint_payload["state_dict"]
    try:
        result = model.load_state_dict(state_dict, strict=strict)
        if hasattr(result, "missing_keys"):
            missing = list(result.missing_keys)
            unexpected = list(result.unexpected_keys)
    except RuntimeError:
        result = model.load_state_dict(state_dict, strict=False)
        if hasattr(result, "missing_keys"):
            missing = list(result.missing_keys)
            unexpected = list(result.unexpected_keys)
    return missing, unexpected
