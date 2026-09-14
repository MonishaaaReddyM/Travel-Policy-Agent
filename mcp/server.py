"""
mcp/server.py
-------------
Day 4 deliverable, Part 2: A basic MCP server exposing the three tools
from src/tools.py, demonstrating tool discovery and invocation over MCP.

    MCP Server
    │
    ├── check_employee_eligibility
    ├── validate_trip
    └── calculate_reimbursement

Per the capstone spec, a production-grade MCP implementation isn't
required here - this demonstrates the core concepts (server, tools, tool
discovery, tool invocation) using the official Python MCP SDK.

Run directly to start the server over stdio:
    python mcp/server.py

To try it with an MCP-aware client (e.g. Claude Desktop or the MCP
Inspector), point the client at this file as a stdio server command:
    python /absolute/path/to/mcp/server.py
"""

import sys
from pathlib import Path

# Make src/ importable so this file can reuse the same tool implementations
# that the agent uses in src/agent.py - one source of truth for the tool
# logic, not a re-implementation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tools import (  # noqa: E402
    check_employee_eligibility as _check_employee_eligibility,
    validate_trip as _validate_trip,
    calculate_reimbursement as _calculate_reimbursement,
)

# The high-level server class was renamed between MCP SDK major versions
# (FastMCP -> MCPServer in mcp 2.x). This import handles either, so this
# file works whether the installed `mcp` package is v1 or v2.
try:
    from mcp.server.fastmcp import FastMCP as _ServerClass
except ModuleNotFoundError:
    from mcp.server.mcpserver import MCPServer as _ServerClass

mcp = _ServerClass("travel-policy-tools")


@mcp.tool()
def check_employee_eligibility(employee_id: str) -> dict:
    """
    Look up an employee's travel eligibility status, country, and employee
    type by employee ID. Returns found=False for an unknown employee ID
    rather than assuming eligibility.
    """
    return _check_employee_eligibility(employee_id).to_dict()


@mcp.tool()
def validate_trip(
    employee_id: str,
    trip_type: str,
    amount: float,
    time: str = "",
    currency: str = "",
) -> dict:
    """
    Validate whether a trip is Approved, Needs Approval, Not Eligible, or
    for an Unknown Employee, based on eligibility status and the country's
    standard spending limit. Also returns the reimbursement breakdown.
    """
    result = _validate_trip(
        employee_id=employee_id,
        trip_type=trip_type,
        amount=amount,
        time=time or None,
        currency=currency or None,
    )
    return result.to_dict()


@mcp.tool()
def calculate_reimbursement(trip_amount: float, policy_limit: float) -> dict:
    """
    Calculate the reimbursable amount (lower of trip_amount and
    policy_limit) and the amount requiring review (any excess).
    """
    return _calculate_reimbursement(trip_amount, policy_limit).to_dict()


if __name__ == "__main__":
    mcp.run()