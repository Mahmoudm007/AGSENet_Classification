from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def _load_clip_components(model_name: str):
    try:
        from transformers import AutoTokenizer, CLIPTextModel
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for CLIP text encoding. "
            "Install the dependencies from requirements.txt before training."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name)
    return tokenizer, model


def _chunk_text(
    text: str,
    tokenizer,
    max_length: int,
    overlap_tokens: int,
) -> List[str]:
    text = " ".join(str(text).strip().split())
    if not text:
        return [""]

    max_content_tokens = max(8, max_length - 2)
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )["input_ids"]
    if len(token_ids) <= max_content_tokens:
        return [text]

    stride = max(1, max_content_tokens - max(0, overlap_tokens))
    chunks: List[str] = []
    for start in range(0, len(token_ids), stride):
        chunk_ids = token_ids[start : start + max_content_tokens]
        if not chunk_ids:
            continue
        decoded = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        chunks.append(decoded or text)
        if start + max_content_tokens >= len(token_ids):
            break
    return chunks or [text]


@torch.no_grad()
def encode_texts_with_clip(
    texts: Sequence[str],
    model_name: str,
    device: torch.device,
    max_length: int = 77,
    batch_size: int = 8,
    overlap_tokens: int = 16,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    tokenizer, model = _load_clip_components(model_name)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    model_max_length = int(getattr(tokenizer, "model_max_length", max_length))
    clip_limit = int(getattr(model.config, "max_position_embeddings", max_length))
    safe_max_length = int(min(max_length, model_max_length, clip_limit))

    flat_chunks: List[str] = []
    owners: List[int] = []
    for idx, text in enumerate(texts):
        for chunk in _chunk_text(text, tokenizer, safe_max_length, overlap_tokens):
            flat_chunks.append(chunk)
            owners.append(idx)

    if not flat_chunks:
        raise ValueError("No class-description texts were provided for CLIP encoding.")

    embeddings: List[torch.Tensor] = []
    for start in range(0, len(flat_chunks), batch_size):
        batch = flat_chunks[start : start + batch_size]
        tokenized = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=safe_max_length,
            verbose=False,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        outputs = model(**tokenized)
        text_features = outputs.pooler_output
        if text_features is None:
            text_features = outputs.last_hidden_state[:, 0, :]
        text_features = F.normalize(text_features, dim=1)
        embeddings.append(text_features.detach().cpu())

    chunk_embeddings = torch.cat(embeddings, dim=0)
    grouped: List[List[torch.Tensor]] = [[] for _ in range(len(texts))]
    for owner, embedding in zip(owners, chunk_embeddings):
        grouped[owner].append(embedding)

    aggregated: List[torch.Tensor] = []
    for per_text in grouped:
        stacked = torch.stack(per_text, dim=0)
        mean_embedding = F.normalize(stacked.mean(dim=0, keepdim=True), dim=1)[0]
        aggregated.append(mean_embedding)

    encoded = torch.stack(aggregated, dim=0).to(torch.float32)
    metadata = {
        "clip_safe_max_length": safe_max_length,
        "clip_num_texts": len(texts),
        "clip_num_chunks": len(flat_chunks),
    }
    return encoded, metadata
