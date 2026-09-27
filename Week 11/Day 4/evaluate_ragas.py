
import asyncio
import json
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

BASE_DIR = Path(__file__).parent
DATASET_FILE = BASE_DIR / "ragas_dataset.json"
RESULTS_FILE = BASE_DIR / "ragas_results.csv"

LLM_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "nomic-embed-text:latest"


async def main():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_FILE}"
        )

    rows = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(rows)} evaluation examples.")

    # Async Ollama-compatible client for the evaluation LLM.
    llm_client = AsyncOpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        timeout=180.0,
        max_retries=1,
    )

    evaluator_llm = llm_factory(
        LLM_MODEL,
        provider="openai",
        client=llm_client,
        adapter="instructor",
    )

    # Async client is required by AnswerRelevancy's aembed_text().
    embedding_client = AsyncOpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        timeout=180.0,
        max_retries=1,
    )

    evaluator_embeddings = embedding_factory(
        "openai",
        model=EMBEDDING_MODEL,
        client=embedding_client,
        interface="modern",
    )

    metrics = [
        ("faithfulness", Faithfulness(llm=evaluator_llm)),
        (
            "answer_relevancy",
            AnswerRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
            ),
        ),
        ("context_precision", ContextPrecision(llm=evaluator_llm)),
        ("context_recall", ContextRecall(llm=evaluator_llm)),
    ]

    results = []

    for index, row in enumerate(rows, start=1):
        print(f"\nEvaluating example {index}/{len(rows)}")

        result_row = {"question": row["question"]}

        for metric_name, metric in metrics:
            if metric_name == "faithfulness":
                metric_inputs = {
                    "user_input": row["question"],
                    "response": row["answer"],
                    "retrieved_contexts": row["contexts"],
                }
            elif metric_name == "answer_relevancy":
                metric_inputs = {
                    "user_input": row["question"],
                    "response": row["answer"],
                }
            else:
                metric_inputs = {
                    "user_input": row["question"],
                    "retrieved_contexts": row["contexts"],
                    "reference": row["ground_truth"],
                }

            result = await metric.ascore(**metric_inputs)
            result_row[metric_name] = result.value
            print(f"{metric_name}: {result.value}")

        results.append(result_row)

        # Save completed examples so progress is retained.
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

    results_df = pd.DataFrame(results)
    print("\nRagas evaluation completed.")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {RESULTS_FILE.name}")


if __name__ == "__main__":
    asyncio.run(main())