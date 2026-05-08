#!/usr/bin/env python3
"""
Interactive chat with the RadOnc RAG Agent.
This connects to real AWS resources (S3, DynamoDB, Bedrock).

Usage:
    python chat.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import create_agent


def main():
    print("=" * 60)
    print("  RadOnc Agentic RAG - Interactive Chat")
    print("=" * 60)
    print("\nInitializing system...")
    print("  - Connecting to AWS...")
    print("  - Loading cross-encoder model...")
    
    try:
        # Create agent with real AWS connection
        agent = create_agent(use_mock=False)
        print("\n System ready!")
        print("  - AWS connected")
        print("  - Cross-encoder model loaded")
        print()
    except Exception as e:
        print(f"ERROR: Failed to connect to AWS: {e}")
        print("\nPlease ensure:")
        print("  1. Run 'aws sso login --profile YOUR_AWS_PROFILE'")
        print("  2. Your AWS credentials are valid")
        return
    
    print("Ask questions about radiation oncology treatment guidelines.")
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'trace' to show reasoning trace for the last query.")
    print("-" * 60)
    
    last_response = None
    
    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if query.lower() == 'trace':
            if last_response:
                print("\n--- Reasoning Trace ---")
                for step in last_response.reasoning_trace:
                    print(f"  {step}")
                print("-" * 30)
            else:
                print("No previous query to show trace for.")
            continue
        
        print("\nThinking...")
        
        try:
            response = agent.query(query)
            last_response = response
            
            # Print answer
            print(f"\nAgent: {response.answer}")
            
            # Print metadata
            print(f"\n[Confidence: {response.confidence_score:.1%} | "
                  f"Citations: {len(response.citations)} | "
                  f"Latency: {response.latency_ms:.0f}ms]")
            
            # Print citations if available
            if response.citations:
                print("\nSources:")
                for c in response.citations[:5]:  # Show top 5
                    print(f"  [{c.citation_id}] {c.source_path}")
                    print(f"      URL: {c.source_url}")
        

            # Print LLM Judge Evaluations if available
            if "judge_evaluations" in response.metadata and response.metadata["judge_evaluations"]:
                ev = response.metadata["judge_evaluations"][0]

                print("\n" + "=" * 70)
                print("                 LLM JUDGE EVALUATION (v3)")
                print("=" * 70)

                # Abstention warning
                if ev.get("abstention_triggered"):
                    print("\n  ⚠️  ABSTENTION TRIGGERED — system should not have answered")
                elif ev.get("should_abstain") == "Yes" and ev.get("did_abstain") == "No":
                    print("\n  ⚠️  WARNING: System answered but evidence was insufficient")

                print(f"\n  Abstention: Should={ev.get('should_abstain','?')} | Did={ev.get('did_abstain','?')}")
                print()

                # Sub-scores with visual bars
                for name, score_key, reasoning_key in [
                    ("Retrieval Adequacy", "retrieval_adequacy", "retrieval_adequacy_reasoning"),
                    ("Faithfulness",       "faithfulness",        "faithfulness_reasoning"),
                    ("Correctness",        "correctness",         "correctness_reasoning"),
                    ("Answer Quality",     "answer_quality",      "answer_quality_reasoning"),
                ]:
                    score = ev.get(score_key, 0)
                    bar   = "█" * score + "░" * (5 - score)
                    print(f"  • {name:<20} {score}/5  [{bar}]")
                    if reasoning_key and ev.get(reasoning_key):
                        print(f"    {ev[reasoning_key]}")
                    print()

                # Composite
                composite = ev.get("composite_score", 0.0)
                pct = (composite / 5.0) * 100
                if composite >= 4.0:
                    print(f"  🎯 Composite Score: {composite:.2f}/5.00 ({pct:.0f}%) — Good")
                elif composite >= 3.0:
                    print(f"  🎯 Composite Score: {composite:.2f}/5.00 ({pct:.0f}%) — Fair")
                elif composite > 0:
                    print(f"  🎯 Composite Score: {composite:.2f}/5.00 ({pct:.0f}%) — Poor")
                else:
                    print(f"  🎯 Composite Score: 0.00/5.00 — Abstention Gate Triggered")

                print("=" * 70)
            

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()