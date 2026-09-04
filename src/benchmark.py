"""Runs the sweep: for each (LLM, embedding model, retrieval strategy),
retrieve and generate an answer for every question, then write per-question
rows and a per-combination summary."""
from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import List

import pandas as pd
import psutil
import yaml
from tqdm import tqdm

from . import dataset as dataset_mod
from . import metrics as metrics_mod
from .embeddings import build_embedder
from .models import build_llm
from .retrieval import build_retriever
from .vectorstore import VectorStore


def _system_info() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
    }


def _build_prompt(question: str, context_docs: List[str]) -> str:
    context = "\n\n".join(context_docs)
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def run_benchmark(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    corpus = dataset_mod.load_corpus(
        cfg["dataset"]["source"],
        n_passages=cfg["dataset"].get("n_passages", 200),
        n_questions=cfg["dataset"].get("n_questions", 100),
        seed=cfg.get("seed", 42),
    ) if cfg["dataset"]["source"] == "squad" else dataset_mod.load_corpus(cfg["dataset"]["source"])

    sys_info = _system_info()
    top_k = cfg["retrieval"]["top_k"]
    system_prompt = cfg["generation"]["system_prompt"]
    llm_options = {
        k: v
        for k, v in {
            "temperature": cfg["generation"].get("temperature"),
            "num_predict": cfg["generation"].get("num_predict"),
        }.items()
        if v is not None
    }
    llm_timeout_s = cfg["generation"].get("timeout_s", 180)

    results_path = Path(cfg["output"]["results_csv"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    done_keys = set()
    if results_path.exists() and results_path.stat().st_size > 0:
        existing = pd.read_csv(results_path)
        rows = existing.to_dict("records")
        done_keys = {
            (r["llm"], r["embedding_model"], r["retrieval_strategy"], r["question"])
            for r in rows
        }
        print(f"Resuming with {len(rows)} existing rows.")

    def _flush():
        pd.DataFrame(rows).to_csv(results_path, index=False)

    def _checkpoint():
        if len(rows) % 10 == 0:
            _flush()

    for embed_cfg in cfg["embeddings"]:
        embedder = build_embedder(embed_cfg["name"], backend=embed_cfg["backend"])

        # Index the corpus once per embedding model, reused by every strategy and LLM.
        vs = VectorStore(
            persist_dir=cfg["vectorstore"]["persist_dir"],
            collection_name=f"corpus-{embed_cfg['name'].replace(':', '_')}",
            reset=not bool(done_keys),
        )
        if vs.count() == 0:
            doc_embeddings = embedder.embed(corpus.documents)
            vs.add(corpus.doc_ids, doc_embeddings, corpus.documents)

        for strategy in cfg["retrieval_strategies"]:
            retriever = build_retriever(
                strategy,
                vectorstore=vs,
                embedder=embedder,
                doc_ids=corpus.doc_ids,
                documents=corpus.documents,
                alpha=cfg["retrieval"].get("hybrid_alpha", 0.5),
            )
            retrieved_by_question = {
                qa.question: retriever.retrieve(qa.question, top_k)
                for qa in corpus.qa_examples
            }

            for llm_cfg in cfg["llms"]:
                pending = [
                    qa
                    for qa in corpus.qa_examples
                    if (llm_cfg["name"], embed_cfg["name"], strategy, qa.question) not in done_keys
                ]
                if not pending:
                    continue

                llm = build_llm(
                    llm_cfg["name"],
                    backend=llm_cfg["backend"],
                    options=llm_options,
                    timeout_s=llm_timeout_s,
                    base_url=llm_cfg.get("base_url"),
                    api_key=llm_cfg.get("api_key", "not-needed"),
                )
                unload_others = getattr(llm, "unload_others", None)
                if unload_others:
                    unload_others({llm_cfg["name"], embed_cfg["name"]})

                try:
                    for qa in tqdm(
                        pending,
                        desc=f"{llm_cfg['name']} | {embed_cfg['name']} | {strategy}",
                    ):
                        retrieved = retrieved_by_question[qa.question]
                        retrieved_ids = [r[0] for r in retrieved]
                        retrieved_texts = [r[1] for r in retrieved]

                        prompt = _build_prompt(qa.question, retrieved_texts)
                        error = None
                        try:
                            gen = llm.generate(prompt, system=system_prompt)
                            prediction = gen.text
                            time_to_first_token_s = gen.time_to_first_token_s
                            total_latency_s = gen.total_latency_s
                            tokens_per_second = gen.tokens_per_second
                            peak_model_mem_mb = gen.peak_model_mem_mb
                            peak_gpu_mem_mb = gen.extra.get("peak_gpu_mem_mb")
                            peak_runner_rss_mb = gen.extra.get("peak_runner_rss_mb")
                            ollama_ps_size_mb = gen.extra.get("ollama_ps_size_mb")
                            prediction_sanitized = bool(gen.extra.get("raw_text"))
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            prediction = ""
                            time_to_first_token_s = None
                            total_latency_s = llm_timeout_s
                            tokens_per_second = None
                            peak_model_mem_mb = None
                            peak_gpu_mem_mb = None
                            peak_runner_rss_mb = None
                            ollama_ps_size_mb = None
                            prediction_sanitized = False

                        rows.append(
                            {
                                "llm": llm_cfg["name"],
                                "embedding_model": embed_cfg["name"],
                                "retrieval_strategy": strategy,
                                "question": qa.question,
                                "gold_answer": qa.answer,
                                "prediction": prediction,
                                "em": metrics_mod.exact_match(prediction, qa.answer),
                                "f1": metrics_mod.f1_score(prediction, qa.answer),
                                "rouge_l": metrics_mod.rouge_l(prediction, qa.answer),
                                "recall_at_k": metrics_mod.recall_at_k(retrieved_ids, qa.gold_passage_id),
                                "mrr": metrics_mod.mrr(retrieved_ids, qa.gold_passage_id),
                                "time_to_first_token_s": time_to_first_token_s,
                                "total_latency_s": total_latency_s,
                                "tokens_per_second": tokens_per_second,
                                "peak_model_mem_mb": peak_model_mem_mb,
                                "peak_gpu_mem_mb": peak_gpu_mem_mb,
                                "peak_runner_rss_mb": peak_runner_rss_mb,
                                "ollama_ps_size_mb": ollama_ps_size_mb,
                                "prediction_sanitized": prediction_sanitized,
                                "generation_error": error,
                                **{f"sys_{k}": v for k, v in sys_info.items()},
                            }
                        )
                        done_keys.add((llm_cfg["name"], embed_cfg["name"], strategy, qa.question))
                        _checkpoint()
                finally:
                    unload = getattr(llm, "unload", None)
                    if unload:
                        unload()
                    _flush()

    results_df = pd.DataFrame(rows)
    Path(cfg["output"]["results_csv"]).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(cfg["output"]["results_csv"], index=False)

    summary_df = (
        results_df.groupby(["llm", "embedding_model", "retrieval_strategy"])
        .agg(
            em=("em", "mean"),
            f1=("f1", "mean"),
            rouge_l=("rouge_l", "mean"),
            recall_at_k=("recall_at_k", "mean"),
            mrr=("mrr", "mean"),
            avg_ttft_s=("time_to_first_token_s", "mean"),
            avg_latency_s=("total_latency_s", "mean"),
            avg_tokens_per_s=("tokens_per_second", "mean"),
            avg_peak_model_mem_mb=("peak_model_mem_mb", "mean"),
            avg_peak_gpu_mem_mb=("peak_gpu_mem_mb", "mean"),
            avg_peak_runner_rss_mb=("peak_runner_rss_mb", "mean"),
        )
        .reset_index()
    )
    summary_df.to_csv(cfg["output"]["summary_csv"], index=False)

    return results_df, summary_df
