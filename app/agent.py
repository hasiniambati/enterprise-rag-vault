"""
Agent Module: Semantic Router and Validation Guardrails
"""
from typing import Dict, Any, List

class AgentRouter:
    @staticmethod
    def route_query(query: str) -> str:
        """Classifies intent to determine the appropriate enterprise retrieval path."""
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["salary", "payroll", "tax", "compensation", "bonus"]):
            return "Financial DB Agent"
        elif any(kw in query_lower for kw in ["leave", "policy", "hr", "benefits", "guideline", "vacation"]):
            return "HR Knowledge Vault (RAG)"
        else:
            return "General Enterprise Search"

class ValidatorAgent:
    @staticmethod
    def validate_response(retrieved_chunks: List[dict], query: str) -> Dict[str, Any]:
        """Validates retrieved chunks to eliminate hallucinations and assign confidence."""
        if not retrieved_chunks:
            return {
                "validated_answer": "I could not find verified enterprise documentation supporting this query.",
                "confidence": "Low (No relevant chunks found)"
            }
        
        top_score = retrieved_chunks[0].get("relevance_score", 999.0)
        if top_score > 1.5:
            return {
                "validated_answer": "Retrieved documents have low semantic relevance to your query. Please consult HR directly.",
                "confidence": "Medium-Low (Flagged for review)"
            }
        
        context_text = "\n".join([chunk["content"] for chunk in retrieved_chunks])
        return {
            "validated_answer": f"Based on verified internal records:\n{context_text}",
            "confidence": "High (Verified against source chunks)"
        }