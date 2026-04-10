"""
Streamlit UI for Northwestern Medicine Agentic RAG System

A modern, professional interface for radiation oncology guideline queries.

Installation:
    pip install streamlit streamlit-extras

Usage:
    streamlit run ui_streamlit.py
"""
import streamlit as st
import sys
import os
import subprocess
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import create_agent, AgenticRAGEngine
from config import default_config
from models import AgentResponse


# Page configuration
st.set_page_config(
    page_title="RadOnc AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4E2A84;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4E2A84;
        margin: 1rem 0;
    }
    .citation-box {
        background-color: #f0f4ff;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .reasoning-step {
        background-color: #f8f9fa;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-left: 3px solid #4E2A84;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    /* metric card styling */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


def check_aws_token() -> bool:
    """Check if AWS SSO token is still valid."""
    result = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--profile", "YOUR_AWS_PROFILE"],
        capture_output=True, text=True
    )
    return result.returncode == 0


def is_token_error(e: Exception) -> bool:
    """Check if an exception is an AWS SSO token expiry error."""
    msg = str(e).lower()
    return "token has expired" in msg or "sso" in msg or "refresh failed" in msg


def initialize_agent():
    """
    Initialize the RAG agent.
    1. Check token via aws sts get-caller-identity
    2. If expired: run aws sso login (opens browser once — expected)
       Show message: AWS session expired, please complete login in browser
    3. If valid or after login: create agent normally
    """
    try:
        # Step 1: Check token
        with st.spinner("🔄 Checking AWS credentials..."):
            token_valid = check_aws_token()

        # Step 2: If expired, run sso login with browser
        if not token_valid:
            st.info("🔐 AWS session expired. Please complete login in your browser...")
            with st.spinner("⏳ Waiting for browser login to complete..."):
                result = subprocess.run(
                    ["aws", "sso", "login", "--profile", "YOUR_AWS_PROFILE"],
                    capture_output=True, text=True, timeout=300
                )
            if result.returncode != 0:
                st.error("❌ AWS SSO login failed or timed out.")
                st.warning("Please run manually in terminal, then click Initialize Agent again:")
                st.code("aws sso login --profile YOUR_AWS_PROFILE")
                return

        # Step 3: Create agent
        with st.spinner("🔄 Connecting to AWS and loading models..."):
            agent = create_agent(use_mock=False)

        st.session_state.initialized = True
        st.session_state.agent = agent
        st.success("✅ Agent initialized successfully!")
        st.rerun()

    except subprocess.TimeoutExpired:
        st.error("❌ Browser login timed out (5 min).")
        st.warning("Please run manually in terminal, then click Initialize Agent again:")
        st.code("aws sso login --profile YOUR_AWS_PROFILE")
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {str(e)}")


def display_response(response: AgentResponse):
    """Display agent response in a formatted way."""

    # Answer section
    st.markdown(f'<div class="answer-box"><h3>💡 Answer</h3>{response.answer}</div>', unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Confidence", f"{response.confidence_score:.1%}")

    with col2:
        st.metric("Citations", len(response.citations))

    with col3:
        st.metric("Latency", f"{response.latency_ms:.0f}ms")

    with col4:
        if response.task_plan:
            st.metric("Tasks", len(response.task_plan.sub_tasks))
        else:
            st.metric("Tasks", "N/A")

    # Citations section
    if response.citations:
        with st.expander(f"📚 View Citations ({len(response.citations)})", expanded=False):
            for citation in response.citations:
                st.markdown(f"""
                <div class="citation-box">
                    <strong>[{citation.citation_id}] {citation.source_path}</strong><br>
                    <em>Relevance: {citation.relevance_score:.2%}</em><br><br>
                    <div style="font-size: 0.9rem; line-height: 1.5;">
                        {citation.chunk_text[:300]}{'...' if len(citation.chunk_text) > 300 else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # LLM Judge Evaluation — reads from pre-computed metadata (no extra API call)
    evaluations = response.metadata.get("judge_evaluations", [])
    if evaluations and evaluations[0].get("overall_score", 0) > 0:
        eval_result = evaluations[0]

        with st.expander("🎯 LLM Judge Evaluation", expanded=True):
            st.caption("Real-time evaluation by Llama 3 70B Judge")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(label="📊 Accuracy", value=f"{eval_result['accuracy']}/5")
                with st.expander("View Reasoning"):
                    st.write(eval_result['accuracy_reasoning'])

            with col2:
                st.metric(label="✅ Completeness", value=f"{eval_result['completeness']}/5")
                with st.expander("View Reasoning"):
                    st.write(eval_result['completeness_reasoning'])

            with col3:
                st.metric(label="🔗 Retrieval Adherence", value=f"{eval_result['retrieval_adherence']}/5")
                with st.expander("View Reasoning"):
                    st.write(eval_result['retrieval_adherence_reasoning'])

            st.markdown("---")
            overall_pct = (eval_result['overall_score'] / 5.0) * 100

            if eval_result['overall_score'] >= 4.0:
                st.success(f"🎯 **Overall Score: {eval_result['overall_score']:.2f}/5.00** ({overall_pct:.0f}%) - Excellent")
            elif eval_result['overall_score'] >= 3.0:
                st.info(f"🎯 **Overall Score: {eval_result['overall_score']:.2f}/5.00** ({overall_pct:.0f}%) - Good")
            else:
                st.warning(f"🎯 **Overall Score: {eval_result['overall_score']:.2f}/5.00** ({overall_pct:.0f}%) - Needs Improvement")

    else:
        st.error("❌ LLM Judge evaluation failed")

    # Reasoning trace section
    with st.expander("🧠 Reasoning Trace", expanded=False):
        for step in response.reasoning_trace:
            st.markdown(f'<div class="reasoning-step">{step}</div>', unsafe_allow_html=True)


def main():
    """Main application."""

    # Header
    st.markdown('<div class="main-header">🏥 RadOnc AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Northwestern Medicine Radiation Oncology Guidelines</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        if not st.session_state.initialized:
            if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
                initialize_agent()
        else:
            st.success("✅ Agent Ready")
            if st.button("🔄 Restart Agent", use_container_width=True):
                st.session_state.initialized = False
                st.session_state.agent = None
                st.rerun()

        st.markdown("---")

        # Sample queries
        #st.markdown("### 💡 Sample Queries")
        #if st.button("Standard dose for breast cancer", key="q1", use_container_width=True):
            #st.session_state.current_query = "What is the standard dose for breast cancer?"
        #if st.button("SBRT vs conventional", key="q2", use_container_width=True):
            #st.session_state.current_query = "Compare SBRT vs conventional fractionation for lung cancer"
        #if st.button("Whole brain radiation side effects", key="q3", use_container_width=True):
            #st.session_state.current_query = "What are the side effects of whole brain radiation?"

    # Main content
    if not st.session_state.agent:
        st.info("👈 Please initialize the agent using the sidebar")
        return

    # Query input
    st.markdown("### 🔍 Ask Your Question")

    default_query = st.session_state.get('current_query', '')

    query = st.text_area(
        "Enter your radiation oncology question:",
        value=default_query,
        height=100,
        placeholder="Example: What is the standard dose fractionation for early-stage breast cancer?"
    )

    if 'current_query' in st.session_state:
        del st.session_state.current_query

    if st.button("🚀 Get Answer", type="primary"):
        if query.strip():
            with st.spinner("Analyzing query and retrieving information..."):
                try:
                    response = st.session_state.agent.query(query)
                    display_response(response)

                    st.markdown("""
                    <div class="warning-box">
                        ⚠️ <strong>Important:</strong> This AI assistant provides information from radiation oncology guidelines.
                        Always verify critical clinical information and consult with the patient's care team.
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    if is_token_error(e):
                        st.error("🔐 AWS session expired.")
                        st.warning('Please click **🔄 Restart Agent** in the sidebar, then **🚀 Initialize Agent** to reconnect.')
                    else:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        with st.expander("View error details"):
                            st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Please enter a question")


if __name__ == "__main__":
    main()