"""Base classes for agentic AI system."""
from typing import List, Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from app.ingestion.loader import Document


class Tool(ABC):
    """Base class for agent tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            input_data: Tool input parameters
            
        Returns:
            Tool execution result
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool input schema."""
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


class AgentState:
    """State maintained by agent during execution."""
    
    def __init__(self, query: str):
        self.query = query
        self.iteration = 0
        self.thoughts: List[str] = []
        self.actions: List[Dict[str, Any]] = []
        self.observations: List[str] = []
        self.final_answer: Optional[str] = None
    
    def add_thought(self, thought: str):
        """Add a thought to the state."""
        self.thoughts.append(thought)
    
    def add_action(self, tool_name: str, input_data: Dict[str, Any]):
        """Record an action."""
        self.actions.append({
            "tool": tool_name,
            "input": input_data,
            "iteration": self.iteration
        })
    
    def add_observation(self, observation: str):
        """Record an observation."""
        self.observations.append(observation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "query": self.query,
            "iteration": self.iteration,
            "thoughts": self.thoughts,
            "actions": self.actions,
            "observations": self.observations,
            "final_answer": self.final_answer
        }


class Agent(ABC):
    """Base agent class."""
    
    def __init__(self, tools: List[Tool], max_iterations: int = 5):
        """
        Initialize agent.
        
        Args:
            tools: List of available tools
            max_iterations: Maximum number of reasoning iterations
        """
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
    
    @abstractmethod
    async def run(self, query: str, context: Optional[str] = None) -> AgentState:
        """
        Run the agent to solve a query.
        
        Args:
            query: User query
            context: Optional initial context
            
        Returns:
            Agent state with execution trace
        """
        pass
    
    async def use_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of the tool
            input_data: Tool input
            
        Returns:
            Tool result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool = self.tools[tool_name]
        return await tool.execute(input_data)

