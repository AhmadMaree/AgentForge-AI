
"""Run BFCL-style evaluation."""

import argparse
import json
import os
from pathlib import Path

from agentforge.evaluation.bfcl import (
    load_cases,
    evaluate,
    save_report,
    markdown_report,
    predict_with_agent,
)

from agentforge.llm.factory import FactoryLLM


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="examples/data/bfcl_sample.json",
    )

    parser.add_argument(
        "--output",
        default="evaluation_results/bfcl_report.json",
    )

    parser.add_argument(
        "--mode",
        choices=["oracle", "gemini"],
        default="oracle",
    )

    args = parser.parse_args()

    cases = load_cases(args.dataset)

    if args.mode == "oracle":
        # Test the pipeline with correct predictions.
        # This is NOT a real model benchmark.
        def predict(case):
            return case["ground_truth"]

    else:
        from agentforge.core.config import LLMConfig
        from agentforge.llm.gemini import GeminiLLM
        from agentforge.agents.simple_agent import (
            SimpleAgent,
        )

        llm = FactoryLLM.create("gemini")

        agent = SimpleAgent(
            "BFCL Evaluation Agent",
            llm,
        )

        def predict(case):
            return predict_with_agent(
                agent,
                case,
            )

    report = evaluate(cases, predict)

    save_report(report, args.output)

    markdown_path = Path(args.output).with_suffix(
        ".md"
    )

    markdown_path.write_text(
        markdown_report(report),
        encoding="utf-8",
    )

    summary = {
        key: value
        for key, value in report.items()
        if key != "results"
    }

    print(json.dumps(summary, indent=2))

    if args.mode == "oracle":
        print(
            "ORACLE SMOKE TEST ONLY: "
            "not a model benchmark score"
        )


if __name__ == "__main__":
    main()
