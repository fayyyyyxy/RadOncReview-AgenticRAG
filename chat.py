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
                print("\n" + "=" * 70)
                print("                 LLM JUDGE EVALUATION")
                print("=" * 70)
                
                evaluations = response.metadata["judge_evaluations"]
                
                for eval in evaluations:
                    print(f"\n🤖 Judge: {eval['model_name']}")
                    print(f"📊 Overall Score: {eval['overall_score']:.2f}/5.00")
                    print()
                    
                    # Print each criterion with visual bar
                    for criterion_name, score_key, reasoning_key in [
                        ("Accuracy", "accuracy", "accuracy_reasoning"),
                        ("Completeness", "completeness", "completeness_reasoning"),
                        ("Retrieval Adherence", "retrieval_adherence", "retrieval_adherence_reasoning")
                    ]:
                        score = eval[score_key]
                        reasoning = eval[reasoning_key]
                        score_bar = "█" * score + "░" * (5 - score)
                        
                        print(f"  • {criterion_name}: {score}/5 [{score_bar}]")
                        print(f"    {reasoning}")
                        print()
                    
                    print(f"  💭 Overall: {eval['overall_reasoning']}")
                    print()
                
                # Calculate and show consensus
                if len(evaluations) > 1:
                    avg_accuracy = sum(e["accuracy"] for e in evaluations) / len(evaluations)
                    avg_completeness = sum(e["completeness"] for e in evaluations) / len(evaluations)
                    avg_retrieval = sum(e["retrieval_adherence"] for e in evaluations) / len(evaluations)
                    avg_overall = sum(e["overall_score"] for e in evaluations) / len(evaluations)
                    
                    print("-" * 70)
                    print("CONSENSUS (Average Across Judges):")
                    print(f"  Accuracy:            {avg_accuracy:.2f}/5.00")
                    print(f"  Completeness:        {avg_completeness:.2f}/5.00")
                    print(f"  Retrieval Adherence: {avg_retrieval:.2f}/5.00")
                    print(f"  Overall:             {avg_overall:.2f}/5.00")
                
                print("=" * 70)
            

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
