
import argparse
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from agentforge.llm.factory import FactoryLLM
from agentforge.memory.embedding import localTransformerEmbedding
from agentforge.rag.expansion import QueryExpander
from agentforge.rag.generator import GroundedGenerator
from agentforge.rag.pipeline import RAGPipeline
from agentforge.rag.store import RAGVectorStore


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--user",
        default="student_001",
    )

    parser.add_argument(
        "--namespace",
        default="learning",
    )

    parser.add_argument("--ingest")
    parser.add_argument("--mqe", action="store_true")
    parser.add_argument("--hyde", action="store_true")

    args = parser.parse_args()

    load_dotenv()

    data_dir = Path("memory_data/rag")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Adapt this call if your existing factory has a
    # different constructor or provider configuration.
    llm = FactoryLLM.create("gemini")
    embedder = localTransformerEmbedding()

    client = QdrantClient(
        path=str(data_dir / "qdrant")
    )

    try:
        store = RAGVectorStore(
            client=client,
            embedder=embedder,
        )

        pipeline = RAGPipeline(
            store=store,
            user_id=args.user,
            namespace=args.namespace,
            expander=QueryExpander(llm),
        )

        if args.ingest:
            count = pipeline.ingest(args.ingest)
            print(f"Indexed {count} chunks.")

        generator = GroundedGenerator(llm)

        print("AgentForge RAG. Enter /exit to quit.")

        while True:
            question = input("\nQuestion: ").strip()

            if question.casefold() in {"/exit", "exit", "quit"}:
                break

            if not question:
                continue

            try:
                hits = pipeline.retrieve(
                    query=question,
                    top_k=5,
                    enable_mqe=args.mqe,
                    enable_hyde=args.hyde,
                )

                result = generator.generate(question, hits)

                print("\nAnswer:")
                print(result.answer)

                if result.citations:
                    print("\nSources:")

                    for citation in result.citations:
                        print(
                            f"[{citation.number}] "
                            f"{citation.source} "
                            f"(page {citation.page})"
                        )

            except Exception as exc:
                print(
                    f"Query failed: {type(exc).__name__}: {exc}"
                )

    finally:
        client.close()


if __name__ == "__main__":
    main()
