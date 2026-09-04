"""Loads the corpus and QA pairs. Defaults to a SQuAD subset; set
dataset.source: custom in the config to use your own documents."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class QAExample:
    question: str
    answer: str
    gold_passage_id: str


@dataclass
class Corpus:
    doc_ids: List[str]
    documents: List[str]
    qa_examples: List[QAExample]


SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SQUAD_DEV_PATH = Path("data") / "squad-dev-v1.1.json"


def _download_squad_dev(dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(SQUAD_DEV_URL, headers={"User-Agent": "local-rag-benchmark"})
    with urllib.request.urlopen(req) as resp:
        dest.write_bytes(resp.read())


def _squad_rows_from_official_json(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for article in payload["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                answers = qa.get("answers") or []
                if not answers:
                    continue
                rows.append(
                    {
                        "context": context,
                        "question": qa["question"],
                        "answers": {"text": [answers[0]["text"]]},
                    }
                )
    return rows


def load_squad_subset(n_passages: int = 200, n_questions: int = 100, seed: int = 42) -> Corpus:
    if not SQUAD_DEV_PATH.exists():
        _download_squad_dev(SQUAD_DEV_PATH)
    ds = _squad_rows_from_official_json(SQUAD_DEV_PATH)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    seen_passages = {}
    qa_examples = []
    for idx in indices:
        row = ds[idx]
        context = row["context"]
        if context not in seen_passages:
            if len(seen_passages) >= n_passages:
                continue
            passage_id = f"squad-{len(seen_passages)}"
            seen_passages[context] = passage_id
        passage_id = seen_passages[context]

        if len(qa_examples) < n_questions and row["answers"]["text"]:
            qa_examples.append(
                QAExample(
                    question=row["question"],
                    answer=row["answers"]["text"][0],
                    gold_passage_id=passage_id,
                )
            )
        if len(seen_passages) >= n_passages and len(qa_examples) >= n_questions:
            break

    doc_ids = list(seen_passages.values())
    documents = list(seen_passages.keys())
    return Corpus(doc_ids=doc_ids, documents=documents, qa_examples=qa_examples)


def load_custom_corpus(data_dir: str = "data") -> Corpus:
    corpus_dir = Path(data_dir) / "corpus"
    qa_path = Path(data_dir) / "qa_pairs.jsonl"

    doc_ids, documents = [], []
    for i, file in enumerate(sorted(corpus_dir.glob("*.txt"))):
        doc_ids.append(f"doc-{i}")
        documents.append(file.read_text(encoding="utf-8"))

    qa_examples = []
    if qa_path.exists():
        with open(qa_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                qa_examples.append(
                    QAExample(
                        question=row["question"],
                        answer=row["answer"],
                        gold_passage_id=row.get("gold_passage_id", ""),
                    )
                )

    return Corpus(doc_ids=doc_ids, documents=documents, qa_examples=qa_examples)


def load_corpus(source: str, **kwargs) -> Corpus:
    if source == "squad":
        return load_squad_subset(**kwargs)
    if source == "custom":
        return load_custom_corpus(**kwargs)
    raise ValueError(f"Unknown dataset source: {source}")
