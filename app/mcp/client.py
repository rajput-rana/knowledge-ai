"""MCP client for connecting to external MCP servers."""
from typing import List, Dict, Any, Optional
import httpx
from app.core.logging import setup_logging

logger = setup_logging()


class MCPClient:
    """Client for connecting to external MCP servers via HTTP."""
    
    def __init__(self, server_name: str, base_url: str):
        """
        Initialize MCP client.
        
        Args:
            server_name: Name of the MCP server
            base_url: Base URL of the MCP server (e.g., "http://localhost:8001")
        """
        self.server_name = server_name
        self.base_url = base_url.rstrip('/')
        self.connected = False
    
    async def connect(self):
        """Connect to the MCP server (verify it's accessible)."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    self.connected = True
                    logger.info(f"Connected to MCP server: {self.server_name}")
                    return True
                else:
                    logger.warning(f"MCP server {self.server_name} returned status {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.server_name}: {e}")
            return False
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/mcp/tools", timeout=10.0)
                response.raise_for_status()
                data = response.json()
                return data.get("tools", [])
        except Exception as e:
            logger.error(f"Error listing tools from {self.server_name}: {e}")
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
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/mcp/tools/{tool_name}",
                    json=arguments,
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                result_text = data.get("result", "")
                return [result_text] if result_text else []
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on {self.server_name}: {e}")
            return [f"Error: {str(e)}"]
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        self.connected = False
        logger.info(f"Disconnected from MCP server: {self.server_name}")


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

