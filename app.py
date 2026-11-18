import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIRECTORY = "./chroma_db"

st.set_page_config(
    page_title="Enterprise Knowledge Vault",
    page_icon="🤖",
    layout="centered"
)

# Custom Dark Theme CSS matching your Tailwind aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #020617;
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] {
        background-color: #090d16;
        border-right: 1px solid #1e293b;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #0f172a !important;
        color: #f1f5f9 !important;
        border: 1px solid #1e293b !important;
        border-radius: 0.5rem !important;
    }
    .stButton button {
        background-color: #4f46e5 !important;
        color: white !important;
        border-radius: 0.5rem !important;
        font-weight: 500;
        border: none;
    }
    .stButton button:hover {
        background-color: #4338ca !important;
    }
    h1, h2, h3, h4 {
        color: #f1f5f9 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)

def ingest_documents(texts, metadatas):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(texts, metadatas=metadatas)
    vectorstore = get_vectorstore()
    vectorstore.add_documents(docs)
    return len(docs)

def similarity_search(query, k=2):
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    formatted = []
    for doc, score in results:
        formatted.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Internal Policy"),
            "relevance_score": round(float(score), 4)
        })
    return formatted

class AgentRouter:
    @staticmethod
    def route_query(query):
        q = query.lower()
        if any(kw in q for kw in ["salary", "payroll", "tax", "bonus"]):
            return "Financial DB Agent"
        elif any(kw in q for kw in ["leave", "policy", "hr", "benefits", "vacation"]):
            return "HR Knowledge Vault (RAG)"
        else:
            return "General Enterprise Search"

class ValidatorAgent:
    @staticmethod
    def validate_response(chunks, query):
        if not chunks:
            return {
                "validated_answer": "I could not find verified enterprise documentation supporting this query.",
                "confidence": "Low (No relevant chunks found)"
            }
        top_score = chunks[0]["relevance_score"]
        if top_score > 1.5:
            return {
                "validated_answer": "Low relevance match. Please consult internal department leads directly.",
                "confidence": "Medium-Low"
            }
        context = "\n".join([c["content"] for c in chunks])
        return {
            "validated_answer": f"Based on verified internal records:\n\n{context}",
            "confidence": "High (Verified against source chunks)"
        }

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Ask Vault", "Ingest Documents"])

if page == "Ask Vault":
    st.title("🛡️ Enterprise Knowledge Vault")
    st.caption("Multi-Agent RAG & Validation System")
    st.markdown("---")

    # Initialize chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your Enterprise Multi-Agent RAG assistant. Ask me questions about company policies, HR guidelines, or ingested documents.",
                "route": None,
                "confidence": None,
                "chunks": []
            }
        ]

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("route"):
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Route Taken:** {message['route']}")
                with col2:
                    st.success(f"**Confidence:** {message['confidence']}")
            
            st.markdown(message["content"])
            
            if message.get("chunks"):
                with st.expander(f"Retrieved Chunks ({len(message['chunks'])})"):
                    for i, c in enumerate(message["chunks"], 1):
                        st.markdown(f"📄 **{c['source']}** (Score: `{c['relevance_score']}`)")
                        st.code(c["content"], language="text")

    # Chat Input at bottom
    if query := st.chat_input("Ask a question (e.g., What is our annual paid leave policy?)..."):
        st.session_state.messages.append({"role": "user", "content": query, "route": None, "confidence": None, "chunks": []})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Agents processing query..."):
                route = AgentRouter.route_query(query)
                chunks = similarity_search(query, k=2)
                validation = ValidatorAgent.validate_response(chunks, query)

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Route Taken:** {route}")
            with col2:
                st.success(f"**Confidence:** {validation['confidence']}")

            st.markdown(validation["validated_answer"])

            if chunks:
                with st.expander(f"Retrieved Chunks ({len(chunks)})"):
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"📄 **{c['source']}** (Score: `{c['relevance_score']}`)")
                        st.code(c["content"], language="text")

            st.session_state.messages.append({
                "role": "assistant",
                "content": validation["validated_answer"],
                "route": route,
                "confidence": validation["confidence"],
                "chunks": chunks
            })

elif page == "Ingest Documents":
    st.title("📂 Ingest New Enterprise Document")
    st.caption("Add policy text and source names to update the vector knowledge base instantly.")
    st.markdown("---")

    doc_text = st.text_area("Document Text Content", height=150, placeholder="Paste company policy or guideline text here...")
    source_name = st.text_input("Source Name / Filename", placeholder="e.g., HR_Policy_2026.txt")

    if st.button("Ingest into Vector Vault"):
        if not doc_text.strip() or not source_name.strip():
            st.error("Please provide both text content and a source name.")
        else:
            try:
                count = ingest_documents([doc_text], [{"source": source_name}])
                st.success(f"Success! Indexed {count} document chunks into ChromaDB.")
            except Exception as e:
                st.error(f"Error ingesting document: {str(e)}")