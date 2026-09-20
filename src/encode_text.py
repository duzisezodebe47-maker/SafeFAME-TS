"""Encode the audited fact corpus once with a frozen, revision-pinned encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def corpus_digest(corpus: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for text_id, fact in zip(corpus["text_id"], corpus["fact"]):
        digest.update(str(text_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(fact).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def run(corpus_path: Path, output: Path, model_name: str, batch_size: int, device: str,
        revision: str = MODEL_REVISION, local_files_only: bool = False) -> np.ndarray:
    corpus = pd.read_csv(corpus_path)
    if corpus["text_id"].duplicated().any() or corpus["fact"].isna().any():
        raise ValueError("Corpus must have unique text_id values and non-empty facts")
    model = SentenceTransformer(model_name, revision=revision, device=device, local_files_only=local_files_only)
    model.eval()
    embeddings = model.encode(
        corpus["fact"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    if embeddings.shape[0] != len(corpus) or not np.allclose(norms, 1.0, atol=1e-4):
        raise AssertionError("Unexpected embedding shape or failed L2 normalization")

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "fact_embeddings.npy", embeddings)
    pd.DataFrame({"row": np.arange(len(corpus)), "text_id": corpus["text_id"]}).to_csv(
        output / "fact_embedding_index.csv", index=False
    )
    metadata = {
        "model": model_name,
        "revision": revision,
        "license": "apache-2.0" if model_name == MODEL_NAME else "check upstream model card",
        "device": device,
        "torch": torch.__version__,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "normalized": True,
        "max_seq_length": model.max_seq_length,
        "batch_size": batch_size,
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": corpus_digest(corpus),
    }
    (output / "fact_embedding_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return embeddings


def verify(output: Path) -> None:
    embeddings = np.load(output / "fact_embeddings.npy", mmap_mode="r")
    index = pd.read_csv(output / "fact_embedding_index.csv")
    metadata = json.loads((output / "fact_embedding_metadata.json").read_text(encoding="utf-8"))
    assert len(index) == embeddings.shape[0] == metadata["shape"][0]
    assert embeddings.shape[1] == metadata["shape"][1]
    assert index["row"].tolist() == list(range(len(index)))
    assert not index["text_id"].duplicated().any()
    corpus_path = Path(metadata['corpus_path'])
    if not corpus_path.exists():
        corpus_path = Path('data_processed/text/fact_corpus.csv')
    corpus = pd.read_csv(corpus_path)
    assert corpus_digest(corpus) == metadata['corpus_sha256'], 'Corpus differs from encoded source'
    assert index.text_id.tolist() == corpus.text_id.tolist(), 'Embedding index differs from corpus order'
    print(f"verified embeddings shape={embeddings.shape} revision={metadata['revision']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data_processed/text/fact_corpus.csv"))
    parser.add_argument("--output", type=Path, default=Path("data_processed/embeddings/all_minilm_l6_v2"))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.verify_only:
        result = run(args.corpus, args.output, args.model, args.batch_size, args.device,
                     args.revision, args.local_files_only)
        print(f"encoded {result.shape[0]} facts into {result.shape[1]} dimensions")
    verify(args.output)
