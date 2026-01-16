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
    print("\nConnecting to AWS...")
    
    try:
        # Create agent with real AWS connection
        agent = create_agent(use_mock=False)
        print("Connected successfully!\n")
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
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
