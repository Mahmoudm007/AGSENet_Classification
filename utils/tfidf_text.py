from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _top_terms_per_text(
    sparse_matrix,
    feature_names: np.ndarray,
    top_k: int,
) -> List[List[Dict[str, float]]]:
    rows: List[List[Dict[str, float]]] = []
    csr = sparse_matrix.tocsr()
    for row_idx in range(csr.shape[0]):
        row = csr.getrow(row_idx)
        if row.nnz == 0:
            rows.append([])
            continue
        data = row.data
        indices = row.indices
        order = np.argsort(data)[::-1][:top_k]
        rows.append(
            [
                {"term": str(feature_names[indices[i]]), "weight": float(data[i])}
                for i in order
            ]
        )
    return rows


def encode_texts_with_tfidf(
    texts: Sequence[str],
    max_features: int = 4000,
    ngram_range: Tuple[int, int] = (1, 2),
    use_svd: bool = False,
    svd_dim: int = 128,
    seed: int = 42,
    top_k_terms: int = 12,
    use_char_ngrams: bool = False,
    char_max_features: int = 2000,
    char_ngram_range: Tuple[int, int] = (3, 5),
    sublinear_tf: bool = True,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    cleaned_texts = [_normalize_text(text) for text in texts]
    if not cleaned_texts:
        raise ValueError("No texts were provided for TF-IDF encoding.")

    word_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=tuple(ngram_range),
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=sublinear_tf,
    )
    word_sparse = word_vectorizer.fit_transform(cleaned_texts)
    word_feature_names = word_vectorizer.get_feature_names_out()
    word_dense = word_sparse.toarray().astype(np.float32)

    def _reduce_features(sparse_matrix, dense_matrix):
        svd_components = None
        if use_svd:
            max_rank = int(min(sparse_matrix.shape[0] - 1, sparse_matrix.shape[1] - 1))
            if max_rank >= 2:
                svd_components = int(min(svd_dim, max_rank))
                reducer = make_pipeline(
                    TruncatedSVD(n_components=svd_components, random_state=seed),
                    Normalizer(copy=False),
                )
                dense_matrix = reducer.fit_transform(sparse_matrix).astype(np.float32)
        return dense_matrix, svd_components

    word_dense, word_svd_components = _reduce_features(word_sparse, word_dense)

    char_sparse = None
    char_dense = None
    char_svd_components = None
    char_feature_names = []
    if use_char_ngrams:
        char_vectorizer = TfidfVectorizer(
            max_features=char_max_features,
            analyzer="char_wb",
            ngram_range=tuple(char_ngram_range),
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=sublinear_tf,
        )
        char_sparse = char_vectorizer.fit_transform(cleaned_texts)
        char_feature_names = char_vectorizer.get_feature_names_out()
        char_dense = char_sparse.toarray().astype(np.float32)
        char_dense, char_svd_components = _reduce_features(char_sparse, char_dense)

    dense_parts = [word_dense]
    if char_dense is not None:
        dense_parts.append(char_dense)
    dense = np.concatenate(dense_parts, axis=1).astype(np.float32)
    row_norm = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = dense / np.clip(row_norm, a_min=1e-8, a_max=None)

    top_terms = _top_terms_per_text(word_sparse, word_feature_names, top_k=top_k_terms)
    similarity = cosine_similarity(dense)

    metadata: Dict[str, object] = {
        "source": "tfidf",
        "tfidf_max_features": int(max_features),
        "tfidf_ngram_range": [int(ngram_range[0]), int(ngram_range[1])],
        "tfidf_sublinear_tf": bool(sublinear_tf),
        "tfidf_vocabulary_size": int(len(word_feature_names)),
        "tfidf_word_vocabulary_size": int(len(word_feature_names)),
        "tfidf_char_vocabulary_size": int(len(char_feature_names)),
        "tfidf_feature_dim": int(dense.shape[1]),
        "tfidf_word_feature_dim": int(word_dense.shape[1]),
        "tfidf_char_feature_dim": int(char_dense.shape[1]) if char_dense is not None else 0,
        "tfidf_char_ngrams_used": bool(use_char_ngrams),
        "tfidf_char_max_features": int(char_max_features),
        "tfidf_char_ngram_range": [int(char_ngram_range[0]), int(char_ngram_range[1])],
        "tfidf_svd_used": bool(word_svd_components is not None or char_svd_components is not None),
        "tfidf_svd_dim_requested": int(svd_dim),
        "tfidf_svd_dim_actual": {
            "word": int(word_svd_components) if word_svd_components is not None else None,
            "char": int(char_svd_components) if char_svd_components is not None else None,
        },
        "top_terms_per_text": top_terms,
        "pairwise_similarity": similarity.astype(np.float32).tolist(),
    }
    return torch.from_numpy(dense), metadata
