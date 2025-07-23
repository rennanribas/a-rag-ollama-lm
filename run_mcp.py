#!/usr/bin/env python3
"""Wrapper script to run MCP server from correct directory."""

import os
import sys
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()

# Change to the project directory
os.chdir(script_dir)

# Add the project directory to Python path
sys.path.insert(0, str(script_dir))

# Import and run the MCP server
if __name__ == "__main__":
    try:
        from src.mcp_server import initialize_components, mcp
        import asyncio
        
        # Initialize components synchronously first
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(initialize_components())
        
        # Run the MCP server
        mcp.run()
    except Exception as e:
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        sys.exit(1)