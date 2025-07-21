"""OpenAI Agents SDK integration for intelligent RAG responses."""

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel
from llama_index.core import Settings

from .config import settings
from .indexer import IncrementalIndexer
from .llm_factory import LLMFactory


class QueryContext(BaseModel):
    """Context for a user query."""
    query: str
    domain: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class AgentResponse(BaseModel):
    """Response from the RAG agent."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query_context: QueryContext
    reasoning: Optional[str] = None
    suggestions: List[str] = []


class RAGAgent:
    """Intelligent RAG agent using configurable LLM providers for query understanding and response generation."""
    
    def __init__(self, indexer: IncrementalIndexer):
        self.indexer = indexer
        
        # Initialize LLM and embedding models using factory
        try:
            self.llm = LLMFactory.create_llm()
            self.embedding_model = LLMFactory.create_embedding_model()
            
            # Configure global LlamaIndex settings
            Settings.llm = self.llm
            Settings.embed_model = self.embedding_model
            
            logger.info(f"Initialized RAG Agent with {settings.llm_provider} LLM and {settings.embedding_provider} embeddings")
            
            # Log provider information
            provider_info = LLMFactory.get_provider_info()
            logger.info(f"Provider status: {provider_info}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM providers: {e}")
            logger.info("Please check your configuration and API keys")
            raise
        
        self.conversation_history: Dict[str, List[Dict]] = {}
        
        # System prompts for different agent functions
        self.system_prompts = {
            "query_analyzer": """
You are a query analysis expert. Your job is to:
1. Understand the user's intent and extract key concepts
2. Identify the domain/topic area
3. Suggest relevant search terms for document retrieval
4. Determine if the query requires code examples, explanations, or specific documentation

Respond with a JSON object containing:
- "intent": brief description of what the user wants
- "domain": the technical domain (e.g., "ios", "swift", "watchos", "api")
- "search_terms": array of relevant search terms
- "query_type": one of ["how_to", "explanation", "example", "reference", "troubleshooting"]
- "complexity": one of ["beginner", "intermediate", "advanced"]
""",
            
            "response_generator": """
You are a technical documentation expert and coding assistant. Your role is to:
1. Provide accurate, helpful responses based on retrieved documentation
2. Include relevant code examples when appropriate
3. Explain concepts clearly for the user's skill level
4. Cite sources and provide links when possible
5. Suggest related topics or next steps

Guidelines:
- Be concise but comprehensive
- Use proper technical terminology
- Include working code examples with explanations
- Highlight important considerations or gotchas
- Structure responses with clear headings and bullet points
- Always cite your sources from the provided documentation
""",
            
            "confidence_assessor": """
You are a response quality assessor. Evaluate the quality and confidence of RAG responses based on:
1. Relevance of retrieved documents to the query
2. Completeness of the answer
3. Accuracy based on source material
4. Clarity and usefulness

Respond with a JSON object containing:
- "confidence": float between 0.0 and 1.0
- "reasoning": brief explanation of the confidence score
- "suggestions": array of follow-up questions or topics
"""
        }
    
    async def analyze_query(self, query_context: QueryContext) -> Dict[str, Any]:
        """Analyze user query to understand intent and extract search terms."""
        try:
            # Use LlamaIndex LLM interface for consistency
            prompt = f"{self.system_prompts['query_analyzer']}\n\nQuery: {query_context.query}"
            response = await self.llm.acomplete(prompt)
            
            analysis_text = str(response)
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = analysis_text[start_idx:end_idx]
                    analysis = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, create a simple analysis
                analysis = {
                    "intent": "General information request",
                    "domain": "general",
                    "search_terms": [query_context.query],
                    "query_type": "explanation",
                    "complexity": "intermediate"
                }
            
            logger.info(f"Query analysis: {analysis.get('intent', 'Unknown')} (domain: {analysis.get('domain', 'general')})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Fallback analysis
            return {
                "intent": "General information request",
                "domain": "general",
                "search_terms": [query_context.query],
                "query_type": "explanation",
                "complexity": "intermediate"
            }
    
    def retrieve_documents(self, search_terms: List[str], top_k: int = 10) -> List[Dict]:
        """Retrieve relevant documents using the indexer."""
        try:
            # Combine search terms into a single query
            combined_query = " ".join(search_terms)
            
            # Query the index
            result = self.indexer.query(combined_query, top_k=top_k)
            
            return result.get("source_documents", [])
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    async def generate_response(self, 
                              query_context: QueryContext, 
                              analysis: Dict[str, Any], 
                              documents: List[Dict]) -> str:
        """Generate a comprehensive response using retrieved documents."""
        try:
            # Prepare context from retrieved documents
            doc_context = "\n\n".join([
                f"Source: {doc['title']} ({doc['url']})\n{doc['content'][:1000]}..."
                for doc in documents[:5]  # Limit to top 5 docs
            ])
            
            # Build conversation context
            session_id = query_context.session_id or "default"
            history = self.conversation_history.get(session_id, [])
            
            messages = [
                {"role": "system", "content": self.system_prompts["response_generator"]}
            ]
            
            # Add conversation history (last 3 exchanges)
            for msg in history[-6:]:
                messages.append(msg)
            
            # Add current query with context
            user_message = f"""
Query: {query_context.query}

Query Analysis:
- Intent: {analysis['intent']}
- Domain: {analysis['domain']}
- Type: {analysis['query_type']}
- Complexity: {analysis['complexity']}

Relevant Documentation:
{doc_context}

Please provide a comprehensive answer based on the retrieved documentation. Include code examples where appropriate and cite your sources.
"""
            
            messages.append({"role": "user", "content": user_message})
            
            # Use LlamaIndex LLM interface
            full_prompt = "\n".join([msg["content"] for msg in messages])
            response = await self.llm.acomplete(full_prompt)
            answer = str(response)
            
            # Update conversation history
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            self.conversation_history[session_id].extend([
                {"role": "user", "content": query_context.query},
                {"role": "assistant", "content": answer}
            ])
            
            # Keep only recent history
            if len(self.conversation_history[session_id]) > 10:
                self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
            
            return answer
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I apologize, but I encountered an error while generating a response. Error: {str(e)}"
    
    async def assess_confidence(self, 
                               query_context: QueryContext, 
                               answer: str, 
                               documents: List[Dict]) -> Dict[str, Any]:
        """Assess the confidence and quality of the generated response."""
        try:
            assessment_prompt = f"""
Query: {query_context.query}
Generated Answer: {answer}

Retrieved Documents Quality:
- Number of documents: {len(documents)}
- Average relevance score: {sum(doc.get('score', 0) for doc in documents) / max(len(documents), 1):.2f}
- Document titles: {[doc.get('title', 'Unknown') for doc in documents[:3]]}

Please assess the quality and confidence of this response.
"""
            
            prompt = f"{self.system_prompts['confidence_assessor']}\n\n{assessment_prompt}"
            response = await self.llm.acomplete(prompt)
            
            assessment_text = str(response)
            # Try to extract JSON from response
            try:
                start_idx = assessment_text.find('{')
                end_idx = assessment_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = assessment_text[start_idx:end_idx]
                    assessment = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError):
                # Fallback assessment based on document scores
                avg_score = sum(doc.get('score', 0) for doc in documents) / max(len(documents), 1)
                assessment = {
                    "confidence": min(avg_score, 0.8),
                    "reasoning": "Automatic assessment based on document relevance scores",
                    "suggestions": ["Try rephrasing your question", "Ask for more specific examples"]
                }
            return assessment
            
        except Exception as e:
            logger.error(f"Confidence assessment failed: {e}")
            # Fallback assessment based on document scores
            avg_score = sum(doc.get('score', 0) for doc in documents) / max(len(documents), 1)
            return {
                "confidence": min(avg_score, 0.8),  # Cap at 0.8 for fallback
                "reasoning": "Automatic assessment based on document relevance scores",
                "suggestions": ["Try rephrasing your question", "Ask for more specific examples"]
            }
    
    async def process_query(self, query_context: QueryContext) -> AgentResponse:
        """Process a complete query through the RAG pipeline."""
        logger.info(f"Processing query: {query_context.query}")
        
        try:
            # Step 1: Analyze the query
            analysis = await self.analyze_query(query_context)
            
            # Step 2: Retrieve relevant documents
            documents = self.retrieve_documents(analysis["search_terms"])
            
            if not documents:
                logger.warning("No relevant documents found")
                return AgentResponse(
                    answer="I couldn't find relevant documentation for your query. Please try rephrasing or check if the documentation has been indexed.",
                    sources=[],
                    confidence=0.1,
                    query_context=query_context,
                    reasoning="No relevant documents retrieved",
                    suggestions=["Try different keywords", "Check if documentation is available"]
                )
            
            # Step 3: Generate response
            answer = await self.generate_response(query_context, analysis, documents)
            
            # Step 4: Assess confidence
            assessment = await self.assess_confidence(query_context, answer, documents)
            
            # Step 5: Prepare final response
            return AgentResponse(
                answer=answer,
                sources=documents,
                confidence=assessment["confidence"],
                query_context=query_context,
                reasoning=assessment["reasoning"],
                suggestions=assessment["suggestions"]
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return AgentResponse(
                answer=f"I encountered an error while processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                query_context=query_context,
                reasoning="Processing error occurred",
                suggestions=["Please try again", "Contact support if the issue persists"]
            )
    
    def clear_conversation_history(self, session_id: Optional[str] = None):
        """Clear conversation history for a session or all sessions."""
        if session_id:
            self.conversation_history.pop(session_id, None)
            logger.info(f"Cleared conversation history for session: {session_id}")
        else:
            self.conversation_history.clear()
            logger.info("Cleared all conversation history")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about conversation history."""
        return {
            "active_sessions": len(self.conversation_history),
            "total_exchanges": sum(len(history) // 2 for history in self.conversation_history.values()),
            "sessions": list(self.conversation_history.keys())
        }