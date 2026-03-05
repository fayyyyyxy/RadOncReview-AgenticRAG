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


def initialize_agent() -> Optional[AgenticRAGEngine]:
    """Initialize the RAG agent."""
    try:
        with st.spinner("🔄 Connecting to AWS and loading models..."):
            agent = create_agent(use_mock=False)
            st.session_state.initialized = True
            st.success("✅ Agent initialized successfully!")
            return agent
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {str(e)}")
        st.info("💡 Check your AWS credentials: aws sso login --profile YOUR_AWS_PROFILE")
        return None


def display_response(response: AgentResponse):
    """Display agent response in a formatted way."""
    
    # Answer section
    # st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    # st.markdown("### 💡 Answer")
    # st.markdown(response.answer)
    # st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="answer-box"><h3>💡 Answer</h3>{response.answer}</div>', unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
       #st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Confidence", f"{response.confidence_score:.1%}")
        #st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Citations", len(response.citations))
        # st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        #st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Latency", f"{response.latency_ms:.0f}ms")
        #st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        #st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if response.task_plan:
            st.metric("Tasks", len(response.task_plan.sub_tasks))
        else:
            st.metric("Tasks", "N/A")
        #st.markdown('</div>', unsafe_allow_html=True)
    
    # Citations section 
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
        
    # LLM Judge Evaluation
    if response.self_critique:
        # Import and run LLM Judge
        from llm_judge import create_llm_judge
        
        # Get retrieved contexts from citations
        retrieved_contexts = [c.chunk_text for c in response.citations] if response.citations else []
        
        # Run LLM Judge
        with st.spinner("Running LLM Judge evaluation..."):
            judge = create_llm_judge(default_config.aws)
            evaluations = judge.evaluate(
                query=response.query,
                answer=response.answer,
                retrieved_contexts=retrieved_contexts
            )
        
        # Display scores from first judge (Llama 3 70B)
        if evaluations and evaluations[0].overall_score > 0:
            eval_result = evaluations[0]
            
            # LLM Judge section as expander
            with st.expander("🎯 LLM Judge Evaluation", expanded=True):
                st.caption("Real-time evaluation by Llama 3 70B Judge")
                
                # Show 3 scores in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="📊 Accuracy",
                        value=f"{eval_result.accuracy.score}/5"
                    )
                    with st.expander("View Reasoning"):
                        st.write(eval_result.accuracy.reasoning)
                
                with col2:
                    st.metric(
                        label="✅ Completeness", 
                        value=f"{eval_result.completeness.score}/5"
                    )
                    with st.expander("View Reasoning"):
                        st.write(eval_result.completeness.reasoning)
                
                with col3:
                    st.metric(
                        label="🔗 Retrieval Adherence",
                        value=f"{eval_result.retrieval_adherence.score}/5"
                    )
                    with st.expander("View Reasoning"):
                        st.write(eval_result.retrieval_adherence.reasoning)
                
                # Overall score
                st.markdown("---")
                overall_pct = (eval_result.overall_score / 5.0) * 100
                
                if eval_result.overall_score >= 4.0:
                    st.success(f"🎯 **Overall Score: {eval_result.overall_score:.2f}/5.00** ({overall_pct:.0f}%) - Excellent")
                elif eval_result.overall_score >= 3.0:
                    st.info(f"🎯 **Overall Score: {eval_result.overall_score:.2f}/5.00** ({overall_pct:.0f}%) - Good")
                else:
                    st.warning(f"🎯 **Overall Score: {eval_result.overall_score:.2f}/5.00** ({overall_pct:.0f}%) - Needs Improvement")
                
                # Overall reasoning
                with st.expander("💭 Overall Assessment"):
                    st.write(eval_result.overall_reasoning)
                    st.caption(f"Judge: {eval_result.model_name} | Time: {eval_result.evaluation_time_ms:.0f}ms")
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
        
        # Initialize agent button
        if not st.session_state.initialized:
            if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
                st.session_state.agent = initialize_agent()
        else:
            st.success("✅ Agent Ready")
            if st.button("🔄 Restart Agent", use_container_width=True):
                st.session_state.initialized = False
                st.session_state.agent = None
                st.rerun()
        
        st.markdown("---")
        
        # Sample queries
        st.markdown("### 💡 Sample Queries")
        if st.button("Standard dose for breast cancer", key="q1", use_container_width=True):
            st.session_state.current_query = "What is the standard dose for breast cancer?"
        if st.button("SBRT vs conventional", key="q2", use_container_width=True):
            st.session_state.current_query = "Compare SBRT vs conventional fractionation for lung cancer"
        if st.button("Whole brain radiation side effects", key="q3", use_container_width=True):
            st.session_state.current_query = "What are the side effects of whole brain radiation?"
        
        st.markdown("---")
        
        # Info
        st.markdown("### ℹ️ System Info")
        st.caption(f"**Model:** {default_config.llm.model_id.split('/')[-1]}")
        st.caption(f"**Region:** {default_config.aws.region}")
    
    # Main content
    if not st.session_state.agent:
        st.info("👈 Please initialize the agent using the sidebar")
        
        # Show medical disclaimer
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Medical Disclaimer:</strong> This AI assistant provides information from radiation oncology guidelines 
            for educational purposes only. All treatment decisions should be made in consultation with qualified healthcare professionals.
        </div>
        """, unsafe_allow_html=True)
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
                    
                    # Medical disclaimer at bottom
                    st.markdown("""
                    <div class="warning-box">
                        ⚠️ <strong>Important:</strong> This AI assistant provides information from radiation oncology guidelines. 
                        Always verify critical clinical information and consult with the patient's care team.
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("View error details"):
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Please enter a question")


if __name__ == "__main__":
    main()
