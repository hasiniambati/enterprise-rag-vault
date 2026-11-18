import streamlit as st
from app.rag import ingest_documents, similarity_search
from app.agent import AgentRouter, ValidatorAgent

# Page Configuration
st.set_page_config(
    page_title="Enterprise Knowledge Vault",
    page_icon="🤖",
    layout="centered"
)

st.title("🛡️ Enterprise Multi-Agent RAG Knowledge Vault")
st.caption("AI-powered agentic retrieval, routing, and validation system.")

# Sidebar Navigation
tabs = st.sidebar.radio("Navigation", ["Ask Vault", "Ingest Documents"])

# TAB 1: ASK VAULT (RAG Chat)
if tabs == "Ask Vault":
    st.subheader("Chat with Enterprise Knowledge Base")
    
    query = st.text_input("Ask a question (e.g., What is our annual paid leave policy?):")
    
    if st.button("Send Query", type="primary"):
        if not query.strip():
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Agents processing query..."):
                # 1. Router Agent
                route = AgentRouter.route_query(query)
                
                # 2. Retrieval Agent
                retrieved_data = similarity_search(query, k=2)
                
                # 3. Validator Agent
                validation = ValidatorAgent.validate_response(retrieved_data, query)
            
            # Display Telemetry Badges
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Route Taken:** {route}")
            with col2:
                st.success(f"**Confidence:** {validation['confidence']}")
            
            # Display Answer
            st.markdown("### Answer")
            st.write(validation["validated_answer"])
            
            # Display Retrieved Chunks Expander
            if retrieved_data:
                with st.expander(f"View Retrieved Chunks ({len(retrieved_data)})"):
                    for i, chunk in enumerate(retrieved_data, 1):
                        st.markdown(f"**Chunk {i} | Source:** `{chunk['source']}` | **Score:** `{chunk['relevance_score']}`")
                        st.text(chunk["content"])

# TAB 2: INGEST DOCUMENTS
elif tabs == "Ingest Documents":
    st.subheader("Ingest New Enterprise Document")
    st.caption("Add policy text and source names to update the vector knowledge base instantly.")
    
    doc_text = st.text_area("Document Text Content", height=150, placeholder="Paste company policy or guideline text here...")
    source_name = st.text_input("Source Name / Filename", placeholder="e.g., HR_Policy_2026.txt")
    
    if st.button("Ingest into Vector Vault", type="primary"):
        if not doc_text.strip() or not source_name.strip():
            st.error("Please provide both text content and a source name.")
        else:
            try:
                metadatas = [{"source": source_name}]
                count = ingest_documents([doc_text], metadatas)
                st.success(f"Success! Indexed {count} document chunks into ChromaDB vector database.")
            except Exception as e:
                st.error(f"Error ingesting document: {str(e)}")