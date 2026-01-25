# LMStack MCP Server

Model Context Protocol (MCP) server for LMStack platform. Enables AI assistants like Claude Desktop, Cursor, and other MCP-compatible clients to interact with your LMStack infrastructure.

## Features

### Resources

| Resource URI | Description |
|-------------|-------------|
| `lmstack://system/status` | Complete system overview |
| `lmstack://workers` | Worker nodes with GPU info |
| `lmstack://containers` | Docker containers |
| `lmstack://deployments` | Model deployments |
| `lmstack://models` | Available models |

### Tools

| Tool | Description |
|------|-------------|
| `get_system_status` | Get complete system status |
| `list_workers` | List all workers with GPU status |
| `list_containers` | List Docker containers |
| `list_deployments` | List model deployments |
| `list_models` | List available models |
| `get_gpu_status` | Get detailed GPU information |
| `deploy_model` | Deploy a model to a worker |
| `stop_deployment` | Stop a running deployment |

## Installation

```bash
cd mcp-server
npm install
npm run build
```

## Configuration

Set environment variables:

```bash
export LMSTACK_API_URL="http://localhost:8000/api"
export LMSTACK_API_TOKEN="your-api-token"
```

## Usage with Claude Desktop

Add to your Claude Desktop config (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "lmstack": {
      "command": "node",
      "args": ["/path/to/lmstack/mcp-server/dist/index.js"],
      "env": {
        "LMSTACK_API_URL": "http://localhost:8000/api",
        "LMSTACK_API_TOKEN": "your-token"
      }
    }
  }
}
```

## Usage with Cursor

Add to Cursor settings:

```json
{
  "mcp.servers": {
    "lmstack": {
      "command": "node",
      "args": ["/path/to/lmstack/mcp-server/dist/index.js"],
      "env": {
        "LMSTACK_API_URL": "http://localhost:8000/api"
      }
    }
  }
}
```

## Development

```bash
# Run in development mode
npm run dev

# Inspect with MCP Inspector
npm run inspect
```

## Example Queries

Once connected, you can ask your AI assistant:

- "Show me the current system status"
- "How much GPU memory is available?"
- "List all running containers"
- "Deploy the Qwen model to worker-1"
- "Stop deployment 5"

## License

MIT
