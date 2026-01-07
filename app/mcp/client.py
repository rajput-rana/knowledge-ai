"""MCP client for connecting to external MCP servers."""
from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from app.core.logging import setup_logging

logger = setup_logging()


class MCPClient:
    """Client for connecting to external MCP servers."""
    
    def __init__(self, server_name: str, command: List[str], args: Optional[List[str]] = None):
        """
        Initialize MCP client.
        
        Args:
            server_name: Name of the MCP server
            command: Command to run the server (e.g., ["python", "server.py"])
            args: Optional arguments for the server
        """
        self.server_name = server_name
        self.command = command
        self.args = args or []
        self.session: Optional[ClientSession] = None
    
    async def connect(self):
        """Connect to the MCP server."""
        try:
            server_params = StdioServerParameters(
                command=self.command[0],
                args=self.command[1:] + self.args
            )
            
            self.session = ClientSession(server_params)
            await self.session.initialize()
            
            logger.info(f"Connected to MCP server: {self.server_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.server_name}: {e}")
            return False
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        if not self.session:
            await self.connect()
        
        try:
            tools = await self.session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                for tool in tools.tools
            ]
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[str]:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            List of text results
        """
        if not self.session:
            await self.connect()
        
        try:
            result = await self.session.call_tool(tool_name, arguments)
            
            # Extract text content
            texts = []
            for content in result.content:
                if hasattr(content, 'text'):
                    texts.append(content.text)
                elif isinstance(content, dict) and 'text' in content:
                    texts.append(content['text'])
            
            return texts
        
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return [f"Error: {str(e)}"]
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.session:
            try:
                await self.session.close()
                logger.info(f"Disconnected from MCP server: {self.server_name}")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
            finally:
                self.session = None


class MCPToolRegistry:
    """Registry for managing multiple MCP clients."""
    
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
    
    def register_client(self, name: str, client: MCPClient):
        """Register an MCP client."""
        self.clients[name] = client
    
    async def get_tool(self, server_name: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific tool from a server."""
        if server_name not in self.clients:
            return None
        
        client = self.clients[server_name]
        tools = await client.list_tools()
        
        for tool in tools:
            if tool["name"] == tool_name:
                return tool
        
        return None
    
    async def call_external_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> List[str]:
        """Call a tool on an external MCP server."""
        if server_name not in self.clients:
            return [f"Error: MCP server '{server_name}' not found"]
        
        client = self.clients[server_name]
        return await client.call_tool(tool_name, arguments)

