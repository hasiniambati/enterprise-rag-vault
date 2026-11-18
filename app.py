import streamlit as st
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIRECTORY = "./chroma_db"

st.set_page_config(
    page_title="Enterprise Knowledge Vault",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Chroma and Embeddings
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
            "validated_answer": f"Based on verified internal records: {context}",
            "confidence": "High (Verified against source chunks)"
        }

# Handle API requests sent from the embedded HTML UI
query_params = st.query_params
if "action" in query_params:
    action = query_params["action"]
    if action == "query":
        q = query_params.get("q", "")
        route = AgentRouter.route_query(q)
        chunks = similarity_search(q, k=2)
        validation = ValidatorAgent.validate_response(chunks, q)
        st.json({
            "route_taken": route,
            "confidence": validation["confidence"],
            "validated_answer": validation["validated_answer"],
            "retrieved_chunks": chunks
        })
        st.stop()
    elif action == "ingest":
        text = query_params.get("text", "")
        source = query_params.get("source", "")
        if text and source:
            count = ingest_documents([text], [{"source": source}])
            st.json({"status": "success", "chunks_indexed": count})
        else:
            st.json({"status": "error", "message": "Missing fields"})
        st.stop()

# Embedded Custom Tailwind UI matching your exact screenshot
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col">
    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-50 px-6 py-4 flex justify-between items-center">
        <div class="flex items-center space-x-3">
            <div class="bg-indigo-600 p-2 rounded-lg font-bold text-white">AI</div>
            <div>
                <h1 class="font-bold text-lg">Enterprise Knowledge Vault</h1>
                <p class="text-xs text-slate-400">Multi-Agent RAG & Validation System</p>
            </div>
        </div>
        <div class="flex space-x-2 bg-slate-800 p-1 rounded-lg">
            <button onclick="switchTab('chat')" id="btn-chat" class="px-4 py-1.5 text-sm font-medium rounded-md bg-indigo-600 text-white transition">Ask Vault</button>
            <button onclick="switchTab('ingest')" id="btn-ingest" class="px-4 py-1.5 text-sm font-medium rounded-md text-slate-400 hover:text-white transition">Ingest Documents</button>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-1 max-w-4xl w-full mx-auto p-6 flex flex-col">
        
        <!-- TAB 1: CHAT INTERFACE -->
        <div id="tab-chat" class="flex-1 flex flex-col space-y-6">
            <div id="chat-container" class="flex-1 space-y-4 overflow-y-auto max-h-[450px] p-4 bg-slate-900/40 rounded-xl border border-slate-800/80">
                <div class="flex items-start space-x-3">
                    <div class="bg-indigo-600 text-xs px-2.5 py-1 rounded-full font-semibold">System</div>
                    <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl text-sm max-w-2xl leading-relaxed">
                        Hello! I am your Enterprise Multi-Agent RAG assistant. Ask me questions about company policies, HR guidelines, or ingested documents.
                    </div>
                </div>
            </div>

            <!-- Query Input -->
            <div class="bg-slate-900 border border-slate-800 p-3 rounded-xl flex items-center space-x-3">
                <input type="text" id="query-input" placeholder="Ask a question (e.g., What is our annual paid leave policy?)..." class="flex-1 bg-transparent border-none outline-none px-3 text-slate-100 placeholder-slate-500 text-sm" onkeypress="if(event.key === 'Enter') sendQuery()">
                <button onclick="sendQuery()" id="send-btn" class="bg-indigo-600 hover:bg-indigo-500 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition">
                    Send Query
                </button>
            </div>
        </div>

        <!-- TAB 2: INGESTION INTERFACE -->
        <div id="tab-ingest" class="hidden flex-1 flex flex-col space-y-6">
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-xl space-y-4">
                <h2 class="text-lg font-semibold">Ingest New Enterprise Document</h2>
                <p class="text-xs text-slate-400">Add plain text documents and source names to update the vector knowledge base instantly.</p>
                
                <div class="space-y-2">
                    <label class="text-xs font-medium text-slate-300">Document Text Content</label>
                    <textarea id="ingest-text" rows="5" class="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm focus:outline-none focus:border-indigo-500 text-slate-100" placeholder="Paste company policy text here..."></textarea>
                </div>

                <div class="space-y-2">
                    <label class="text-xs font-medium text-slate-300">Source Name / Filename</label>
                    <input type="text" id="ingest-source" class="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm focus:outline-none focus:border-indigo-500 text-slate-100" placeholder="e.g., HR_Policy_2026.txt">
                </div>

                <button onclick="ingestDocument()" id="ingest-btn" class="bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-2.5 rounded-lg text-sm font-medium transition">
                    Ingest into Vector Vault
                </button>
                <div id="ingest-status" class="text-xs font-medium"></div>
            </div>
        </div>
    </main>

    <script>
        function switchTab(tab) {
            const chatTab = document.getElementById('tab-chat');
            const ingestTab = document.getElementById('tab-ingest');
            const btnChat = document.getElementById('btn-chat');
            const btnIngest = document.getElementById('btn-ingest');

            if (tab === 'chat') {
                chatTab.classList.remove('hidden');
                ingestTab.classList.add('hidden');
                btnChat.className = "px-4 py-1.5 text-sm font-medium rounded-md bg-indigo-600 text-white transition";
                btnIngest.className = "px-4 py-1.5 text-sm font-medium rounded-md text-slate-400 hover:text-white transition";
            } else {
                ingestTab.classList.remove('hidden');
                chatTab.classList.add('hidden');
                btnIngest.className = "px-4 py-1.5 text-sm font-medium rounded-md bg-indigo-600 text-white transition";
                btnChat.className = "px-4 py-1.5 text-sm font-medium rounded-md text-slate-400 hover:text-white transition";
            }
        }

        async function sendQuery() {
            const input = document.getElementById('query-input');
            const container = document.getElementById('chat-container');
            const query = input.value.trim();
            if (!query) return;

            container.innerHTML += `
                <div class="flex items-start justify-end space-x-3">
                    <div class="bg-indigo-600/20 border border-indigo-500/30 p-3 rounded-xl text-sm max-w-2xl leading-relaxed text-slate-100">
                        ${query}
                    </div>
                    <div class="bg-slate-700 text-xs px-2.5 py-1 rounded-full font-semibold">You</div>
                </div>
            `;
            input.value = '';
            container.scrollTop = container.scrollHeight;

            try {
                const res = await fetch(`/?action=query&q=${encodeURIComponent(query)}`);
                const data = await res.json();

                let chunksHtml = '';
                if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
                    chunksHtml = `<div class="mt-3 pt-3 border-t border-slate-800 text-xs space-y-1 text-slate-400">
                        <span class="font-semibold text-slate-300">Retrieved Chunks (${data.retrieved_chunks.length}):</span>
                        ${data.retrieved_chunks.map(c => `<div class="bg-slate-950 p-2 rounded border border-slate-800">📄 <b>${c.source}</b> (Score: ${c.relevance_score}): ${c.content}</div>`).join('')}
                    </div>`;
                }

                container.innerHTML += `
                    <div class="flex items-start space-x-3">
                        <div class="bg-indigo-600 text-xs px-2.5 py-1 rounded-full font-semibold">Agent</div>
                        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl text-sm max-w-2xl leading-relaxed space-y-2">
                            <div class="flex space-x-2">
                                <span class="bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 text-[10px] px-2 py-0.5 rounded font-mono">Route: ${data.route_taken}</span>
                                <span class="bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 text-[10px] px-2 py-0.5 rounded font-mono">Confidence: ${data.confidence}</span>
                            </div>
                            <p class="text-slate-200">${data.validated_answer}</p>
                            ${chunksHtml}
                        </div>
                    </div>
                `;
            } catch (err) {
                container.innerHTML += `<div class="text-red-400 text-xs">Error connecting to RAG backend.</div>`;
            }
            container.scrollTop = container.scrollHeight;
        }

        async function ingestDocument() {
            const text = document.getElementById('ingest-text').value.trim();
            const source = document.getElementById('ingest-source').value.trim();
            const statusDiv = document.getElementById('ingest-status');

            if (!text || !source) {
                statusDiv.className = "text-xs font-medium text-red-400";
                statusDiv.innerText = "Please provide both text content and a source name.";
                return;
            }

            try {
                const res = await fetch(`/?action=ingest&text=${encodeURIComponent(text)}&source=${encodeURIComponent(source)}`);
                const data = await res.json();
                statusDiv.className = "text-xs font-medium text-emerald-400";
                statusDiv.innerText = `Success! Indexed ${data.chunks_indexed} document chunks into ChromaDB.`;
                document.getElementById('ingest-text').value = '';
                document.getElementById('ingest-source').value = '';
            } catch (err) {
                statusDiv.className = "text-xs font-medium text-red-400";
                statusDiv.innerText = "Error ingesting document.";
            }
        }
    </script>
</body>
</html>
"""

# Render the exact custom Tailwind UI inside Streamlit
st.components.v1.html(html_code, height=750, scrolling=True)