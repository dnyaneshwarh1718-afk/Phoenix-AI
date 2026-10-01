from app.rag.generation.answer_generator import (
    AnswerGenerator,
)

from app.rag.generation.prompt_builder import (
    PromptBuilder,
)


def main():

    print("=" * 70)
    print("PHOENIX AI - ANSWER GENERATION TEST")
    print("=" * 70)

    query = (
        "What component controls the specialized agents?"
    )

    context = """
PHOENIX AI ARCHITECTURE

Phoenix AI is a modular agentic AI system designed
to coordinate multiple specialized AI agents.

The Orchestrator Agent is the central controller
of Phoenix AI. It receives user requests, determines
the required task, and delegates work to specialized
agents.

The RAG Agent is responsible for document ingestion,
document retrieval, context construction, and
answering questions using information contained
in user documents.

The Research Agent performs external research and
gathers information from available research sources.

The Planning Agent decomposes complex user requests
into multiple executable steps and coordinates task
execution.

The Memory Agent manages persistent and conversational
memory for Phoenix AI.

The Application Control Agent interacts with approved
desktop applications such as Excel, PowerPoint,
Word, Power BI, Jupyter Notebook, and other supported
applications.
"""

    print(
        f"\nQuery: {query}"
    )

    print(
        "\n" + "=" * 70
    )

    print("GENERATING ANSWER")

    print("=" * 70)

    prompt_builder = PromptBuilder()

    generator = AnswerGenerator(
        model_name="qwen3:4b-instruct"
    )

    result = generator.generate_from_context(
        query=query,
        context=context,
        prompt_builder=prompt_builder,
    )

    print(
        f"\nProvider: {result.provider}"
    )

    print(
        f"Model: {result.model}"
    )

    print(
        "\nAnswer:"
    )

    print(
        result.answer
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ANSWER GENERATION TEST COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()