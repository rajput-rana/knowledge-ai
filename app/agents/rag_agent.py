"""RAG agent that uses tools to answer questions."""
from typing import Optional, List, Dict, Any
from app.agents.base import Agent, AgentState, Tool
from app.llm.chat import ChatCompletion
from app.prompts.templates import PromptTemplate, PromptStyle
from app.core.config import Settings


class RAGAgent(Agent):
    """Agent that uses RAG tools to answer questions."""
    
    def __init__(
        self,
        tools: List[Tool],
        chat_completion: ChatCompletion,
        settings: Settings,
        max_iterations: int = 3
    ):
        """
        Initialize RAG agent.
        
        Args:
            tools: List of available tools
            chat_completion: Chat completion client
            settings: Application settings
            max_iterations: Maximum iterations
        """
        super().__init__(tools, max_iterations)
        self.chat_completion = chat_completion
        self.settings = settings
    
    async def run(self, query: str, context: Optional[str] = None) -> AgentState:
        """
        Run agent to answer query.
        
        Args:
            query: User query
            context: Optional initial context
            
        Returns:
            Agent state with answer and execution trace
        """
        state = AgentState(query)
        
        # Initial thought
        state.add_thought(f"I need to answer: {query}")
        
        # If no initial context, retrieve information
        if not context:
            state.add_thought("No initial context provided. I'll search the knowledge base.")
            
            # Use RAG retrieval tool
            if "rag_retrieve" in self.tools:
                state.iteration += 1
                state.add_action("rag_retrieve", {"query": query})
                
                result = await self.use_tool("rag_retrieve", {"query": query})
                state.add_observation(f"Retrieved {result.get('count', 0)} relevant documents")
                
                # Build context from retrieved documents
                retrieved_docs = result.get("results", [])
                if retrieved_docs:
                    context_parts = []
                    for doc in retrieved_docs[:5]:  # Limit to top 5
                        context_parts.append(f"[Document ID: {doc['doc_id']}]\n{doc['text']}\n")
                    context = "\n---\n\n".join(context_parts)
                else:
                    context = "No relevant documents found in the knowledge base."
        
        # Generate answer using LLM with context
        state.add_thought("I have the context. Now I'll generate a comprehensive answer.")
        
        # Build prompt with chain-of-thought style
        messages = PromptTemplate.build_prompt(
            query=query,
            context=context or "No context available.",
            style=PromptStyle.CHAIN_OF_THOUGHT,
            include_examples=False
        )
        
        # Add agent reasoning to the prompt
        agent_prompt = f"""You are an AI agent that has access to tools and retrieved information.

Retrieved Context:
{context or "No context available."}

Your task: Answer the following question using the context above.

Think step by step:
1. What information do I have?
2. What information do I need?
3. How can I synthesize this information?
4. What is my final answer?

Question: {query}

Let's think step by step:"""
        
        messages = [
            {"role": "system", "content": PromptTemplate.CHAIN_OF_THOUGHT_SYSTEM_PROMPT},
            {"role": "user", "content": agent_prompt}
        ]
        
        answer = await self.chat_completion.complete(messages, temperature=0.7)
        state.final_answer = answer
        state.add_observation("Generated answer using LLM")
        
        return state
    
    async def run_with_tool_use(self, query: str) -> AgentState:
        """
        Run agent with explicit tool use decisions.
        
        Args:
            query: User query
            
        Returns:
            Agent state with answer
        """
        state = AgentState(query)
        
        # Use LLM to decide which tools to use
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        decision_prompt = f"""You are an AI agent with access to these tools:

{tool_descriptions}

Question: {query}

What tools should I use to answer this question? Think step by step."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that decides which tools to use."},
            {"role": "user", "content": decision_prompt}
        ]
        
        decision = await self.chat_completion.complete(messages, temperature=0.3)
        state.add_thought(f"Tool selection reasoning: {decision}")
        
        # For now, always use RAG retrieval
        if "rag_retrieve" in self.tools:
            state.iteration += 1
            state.add_action("rag_retrieve", {"query": query})
            
            result = await self.use_tool("rag_retrieve", {"query": query})
            state.add_observation(f"Retrieved {result.get('count', 0)} documents")
            
            # Generate answer
            retrieved_docs = result.get("results", [])
            if retrieved_docs:
                context = "\n---\n\n".join([
                    f"[{doc['doc_id']}]\n{doc['text']}"
                    for doc in retrieved_docs[:5]
                ])
            else:
                context = "No relevant documents found."
            
            answer_messages = PromptTemplate.build_prompt(
                query=query,
                context=context,
                style=PromptStyle.CHAIN_OF_THOUGHT
            )
            
            answer = await self.chat_completion.complete(answer_messages, temperature=0.7)
            state.final_answer = answer
        
        return state

