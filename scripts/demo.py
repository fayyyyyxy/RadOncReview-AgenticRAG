"""
Demo and testing script for the Agentic RAG Engine.
Run this to test the system without AWS dependencies.
"""
import sys
import os
import json
from datetime import datetime

# Add project root to path when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import create_agent
from config import Config, default_config


def print_separator(title: str = ""):
    """Print a visual separator."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)
    else:
        print('-'*60)


def format_response(response) -> str:
    """Format agent response for display."""
    output = []

    output.append(f"\n Query: {response.query}")
    output.append(f"\n Answer:\n{response.answer}")
    output.append(f"\n Confidence: {response.confidence_score:.2%}")

    if response.citations:
        output.append(f"\n Citations ({len(response.citations)}):")
        for citation in response.citations:
            output.append(f"  [{citation.citation_id}] {citation.source_path}")
            output.append(f"      Relevance: {citation.relevance_score:.2f}")

    if response.self_critique:
        critique = response.self_critique
        output.append(f"\n Self-Critique:")
        output.append(f"  Complete: {'✓' if critique.is_complete else '✗'}")
        output.append(f"  Accurate: {'✓' if critique.is_accurate else '✗'}")
        output.append(f"  Safe: {'✓' if critique.is_safe else '✗'}")
        if critique.missing_aspects:
            output.append(f"  Missing: {', '.join(critique.missing_aspects)}")

    output.append(f"\n Latency: {response.latency_ms:.0f}ms")

    return '\n'.join(output)


def demo_query_analysis():
    """Demo the query analysis component."""
    print_separator("Query Analysis Demo")

    from core.query_analyzer import QueryAnalyzer
    from core.llm_client import MockLLMClient

    analyzer = QueryAnalyzer(MockLLMClient(), default_config)

    test_queries = [
        "What is the standard dose for breast cancer IMRT?",
        "Compare SBRT vs conventional fractionation for early lung cancer",
        "What are the side effects of whole brain radiation?",
        "T2N1M0 breast cancer staging and treatment options",
        "When is proton therapy indicated for pediatric brain tumors?",
    ]

    for query in test_queries:
        analysis = analyzer.analyze(query)
        print(f"\nQuery: {query}")
        print(f"  Intent: {analysis.intent.value}")
        print(f"  Disease: {analysis.disease_category}")
        print(f"  Modality: {analysis.treatment_modality}")
        print(f"  Entities: {analysis.key_entities}")
        print(f"  Complex: {analysis.is_complex}")
        print(f"  Confidence: {analysis.confidence:.2f}")


def demo_task_planning():
    """Demo the task planning component."""
    print_separator("Task Planning Demo")

    from core.query_analyzer import QueryAnalyzer
    from core.task_planner import TaskPlanner
    from core.llm_client import MockLLMClient

    llm = MockLLMClient()
    analyzer = QueryAnalyzer(llm, default_config)
    planner = TaskPlanner(llm, default_config)

    test_queries = [
        "What is the dose for breast cancer?",  # Simple
        "Compare IMRT vs VMAT for prostate cancer",  # Comparison
        "What are the indications, contraindications, and dose for SBRT in lung cancer?",  # Complex
    ]

    for query in test_queries:
        analysis = analyzer.analyze(query)
        plan = planner.plan(analysis)

        print(f"\nQuery: {query}")
        print(f"Plan ID: {plan.plan_id}")
        print(f"Reasoning: {plan.reasoning}")
        print(f"Sub-tasks ({len(plan.sub_tasks)}):")
        for task in plan.sub_tasks:
            print(f"  - {task.description}")
            print(f"    Query: {task.query}")
            print(f"    Strategy: {task.retrieval_strategy.value}")


def demo_guardrails():
    """Demo the guardrails component."""
    print_separator("Guardrails Demo")

    from core.guardrails import GuardrailManager

    guardrails = GuardrailManager(default_config.guardrail)

    test_inputs = [
        "What is the dose for breast cancer?",  # Clean
        "Patient John Smith, DOB 01/15/1980, MRN 12345, needs treatment",  # PHI
        "What is the lethal dose of radiation?",  # Potentially harmful
        "A" * 3000,  # Too long
    ]

    print("\nInput Guardrails:")
    for text in test_inputs:
        result = guardrails.check_input(text[:100] + "..." if len(text) > 100 else text)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n  {status}: {text[:60]}...")
        if result.warnings:
            for warning in result.warnings:
                print(f"    ⚠️ {warning}")

    test_outputs = [
        "The recommended dose is 50 Gy in 25 fractions.",  # Clean
        "Stop taking your medication immediately.",  # Unsafe
        "This treatment will definitely cure your cancer.",  # Absolute claim
    ]

    print("\nOutput Guardrails:")
    for text in test_outputs:
        result = guardrails.check_output(text)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n  {status}: {text[:60]}...")
        if result.warnings:
            for warning in result.warnings:
                print(f"    ⚠️ {warning}")


def demo_full_pipeline():
    """Demo the full agentic RAG pipeline with mock components."""
    print_separator("Full Pipeline Demo (Mock Mode)")

    # Create agent in mock mode
    agent = create_agent(use_mock=True)

    test_queries = [
        "What is the standard dose fractionation for early-stage breast cancer?",
        "Compare hypofractionation vs conventional fractionation for breast cancer",
        "What are the indications for SBRT in lung cancer?",
    ]

    for query in test_queries:
        print_separator()
        response = agent.query(query)
        print(format_response(response))

        # Show reasoning trace
        print("\n Reasoning Trace:")
        for step in response.reasoning_trace[-5:]:  # Last 5 steps
            print(f"  {step}")


def demo_interactive():
    """Interactive demo mode."""
    print_separator("Interactive Demo Mode")
    print("Type your radiation oncology questions (or 'quit' to exit)")
    print("Note: Running in mock mode - responses are simulated")

    agent = create_agent(use_mock=True)

    while True:
        print("\n" + "-"*40)
        query = input("\n Your question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            continue

        print("\nProcessing...")
        response = agent.query(query)
        print(format_response(response))


def run_all_demos():
    """Run all demonstration functions."""
    demo_query_analysis()
    demo_task_planning()
    demo_guardrails()
    demo_full_pipeline()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agentic RAG Engine Demo")
    parser.add_argument(
        "--mode",
        choices=["all", "analysis", "planning", "guardrails", "pipeline", "interactive"],
        default="all",
        help="Demo mode to run"
    )

    args = parser.parse_args()

    print("\n" + "RadOnc Agentic RAG Engine Demo".center(60))
    print("="*60)

    if args.mode == "all":
        run_all_demos()
    elif args.mode == "analysis":
        demo_query_analysis()
    elif args.mode == "planning":
        demo_task_planning()
    elif args.mode == "guardrails":
        demo_guardrails()
    elif args.mode == "pipeline":
        demo_full_pipeline()
    elif args.mode == "interactive":
        demo_interactive()
