"""Mock BeeAI tools with deliberate traps for benchmarking evolving-memory.

These tools simulate realistic failure scenarios that small LLMs encounter:
- Wrong column names in SQL databases
- Missing library imports in Python execution
- Hidden pagination tokens in API responses
- Stripe API with amounts in cents and cursor pagination in headers

Usage:
    from mock_tools import sales_db_query, python_calc, paginated_api_fetch, reset_api_state
    from mock_tools import stripe_charges_api, reset_stripe_state
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr

from beeai_framework.tools import StringToolOutput, tool


# ── Tool 1: Sales Database (SQL Trap) ────────────────────────────────

_DB_CONN: sqlite3.Connection | None = None


def _get_sales_db() -> sqlite3.Connection:
    """Create and populate the sales database with Spanish column names."""
    global _DB_CONN
    if _DB_CONN is not None:
        return _DB_CONN

    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE ventas_q3 (
            id INTEGER PRIMARY KEY,
            categoria TEXT NOT NULL,
            semana INTEGER NOT NULL,
            ingresos_centavos INTEGER NOT NULL,
            unidades_vendidas INTEGER NOT NULL
        )
    """)

    # Insert realistic sales data for "Electronica" category
    data = [
        ("Electronica", 1, 125000, 50),
        ("Electronica", 2, 137500, 55),
        ("Electronica", 3, 148750, 59),
        ("Electronica", 4, 156200, 62),
        ("Electronica", 5, 171800, 68),
        ("Electronica", 6, 189000, 75),
        ("Electronica", 7, 196350, 78),
        ("Electronica", 8, 215000, 85),
        ("Electronica", 9, 210700, 83),
        ("Electronica", 10, 232000, 92),
        ("Electronica", 11, 248500, 98),
        ("Electronica", 12, 268200, 106),
        # Other categories for realism
        ("Ropa", 1, 85000, 120),
        ("Ropa", 2, 87000, 125),
        ("Ropa", 3, 91000, 130),
        ("Hogar", 1, 65000, 40),
        ("Hogar", 2, 68000, 42),
        ("Hogar", 3, 72000, 45),
    ]
    conn.executemany(
        "INSERT INTO ventas_q3 (categoria, semana, ingresos_centavos, unidades_vendidas) VALUES (?, ?, ?, ?)",
        data,
    )
    conn.commit()
    _DB_CONN = conn
    return conn


@tool
def sales_db_query(query: str) -> StringToolOutput:
    """Execute a SQL query against the Q3 sales database.

    Available tables:
      - ventas_q3: Contains quarterly sales data with revenue and units sold

    Args:
        query: A valid SQL query to execute against the database.

    Returns:
        Query results as formatted text, or an error message if the query fails.
    """
    conn = _get_sales_db()
    try:
        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()

        if not rows:
            return StringToolOutput("Query returned 0 rows.")

        # Format as readable table
        lines = [" | ".join(columns)]
        lines.append("-" * len(lines[0]))
        for row in rows[:50]:  # Limit output
            lines.append(" | ".join(str(v) for v in row))

        result = "\n".join(lines)
        if len(rows) > 50:
            result += f"\n... ({len(rows)} total rows, showing first 50)"
        return StringToolOutput(result)

    except sqlite3.OperationalError as e:
        return StringToolOutput(f"SQL Error: {e}")
    except Exception as e:
        return StringToolOutput(f"Error: {e}")


# ── Tool 2: Python Calculator (Import Trap) ──────────────────────────

@tool
def python_calc(code: str) -> StringToolOutput:
    """Execute Python code for data analysis and calculations.

    The execution environment has access to Python's standard library only.
    External packages like pandas, numpy, scipy are NOT available.
    Use built-in functions, math module, statistics module, or manual calculations.

    Args:
        code: Python code to execute. Use print() to output results.

    Returns:
        The stdout output from code execution, or error details if it fails.
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Restricted namespace — standard library only
    namespace: dict = {"__builtins__": __builtins__}

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        if errors:
            return StringToolOutput(f"Output:\n{output}\nWarnings:\n{errors}")
        if output:
            return StringToolOutput(output)
        return StringToolOutput("Code executed successfully (no output printed).")

    except ImportError as e:
        return StringToolOutput(
            f"ImportError: {e}\n"
            f"Note: Only Python standard library is available. "
            f"Use built-in functions, math, statistics, or csv modules instead."
        )
    except Exception as e:
        tb = traceback.format_exc()
        return StringToolOutput(f"Error:\n{tb}")


# ── Tool 3: Paginated API (Header Trap) ──────────────────────────────

# Simulated employee database
_EMPLOYEES = [
    {"id": 1, "name": "Ana Garcia", "department": "Engineering", "salary": 95000},
    {"id": 2, "name": "Carlos Lopez", "department": "Marketing", "salary": 72000},
    {"id": 3, "name": "Maria Rodriguez", "department": "Engineering", "salary": 105000},
    {"id": 4, "name": "Pedro Martinez", "department": "Sales", "salary": 68000},
    {"id": 5, "name": "Laura Sanchez", "department": "Engineering", "salary": 98000},
    {"id": 6, "name": "Juan Hernandez", "department": "HR", "salary": 65000},
    {"id": 7, "name": "Sofia Diaz", "department": "Engineering", "salary": 110000},
    {"id": 8, "name": "Diego Torres", "department": "Marketing", "salary": 75000},
    {"id": 9, "name": "Valentina Ruiz", "department": "Sales", "salary": 71000},
    {"id": 10, "name": "Mateo Flores", "department": "Engineering", "salary": 102000},
    {"id": 11, "name": "Camila Vargas", "department": "HR", "salary": 67000},
    {"id": 12, "name": "Andres Castro", "department": "Engineering", "salary": 115000},
    {"id": 13, "name": "Isabella Morales", "department": "Marketing", "salary": 78000},
]

_PAGE_SIZE = 5
_PAGE_TOKENS = {
    "": "page_alpha",       # First page → next token
    "page_alpha": "page_beta",  # Second page → next token
    "page_beta": "",        # Third page → no more
}

_api_call_count = 0


def reset_api_state() -> None:
    """Reset the API call counter between demo runs."""
    global _api_call_count
    _api_call_count = 0


@tool
def paginated_api_fetch(endpoint: str) -> StringToolOutput:
    """Fetch data from the Employee Directory API.

    Base endpoint: /api/employees
    The API returns paginated results. Check the full response including
    headers for pagination information.

    Args:
        endpoint: API endpoint path, e.g. "/api/employees" or "/api/employees?param=value"

    Returns:
        Full HTTP response including headers and JSON body.
    """
    global _api_call_count
    _api_call_count += 1

    # Parse the endpoint
    path = endpoint.split("?")[0] if "?" in endpoint else endpoint
    params: dict[str, str] = {}
    if "?" in endpoint:
        param_str = endpoint.split("?", 1)[1]
        for pair in param_str.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                params[k] = v

    # Validate endpoint
    if path.rstrip("/") != "/api/employees":
        return StringToolOutput(json.dumps({
            "status": 404,
            "headers": {"Content-Type": "application/json"},
            "body": {"error": "Not found", "message": f"Unknown endpoint: {path}"},
        }, indent=2))

    # Determine page token
    page_token = params.get("page_token", "")

    # Common mistakes: trying page=N, offset=N, skip=N, cursor=N
    for bad_param in ["page", "offset", "skip", "cursor", "start", "limit"]:
        if bad_param in params:
            return StringToolOutput(json.dumps({
                "status": 400,
                "headers": {"Content-Type": "application/json"},
                "body": {
                    "error": "Bad Request",
                    "message": f"Unknown parameter '{bad_param}'. "
                               f"This API uses token-based pagination.",
                },
            }, indent=2))

    # Validate token
    if page_token not in _PAGE_TOKENS:
        return StringToolOutput(json.dumps({
            "status": 400,
            "headers": {"Content-Type": "application/json"},
            "body": {
                "error": "Invalid page token",
                "message": f"Token '{page_token}' is not valid.",
            },
        }, indent=2))

    # Determine slice
    if page_token == "":
        start = 0
    elif page_token == "page_alpha":
        start = 5
    else:  # page_beta
        start = 10

    page_data = _EMPLOYEES[start : start + _PAGE_SIZE]
    next_token = _PAGE_TOKENS[page_token]
    has_more = bool(next_token)

    # Build response — the TRAP: next_page_token is in HEADERS, not body
    response = {
        "status": 200,
        "headers": {
            "Content-Type": "application/json",
            "X-Total-Count": str(len(_EMPLOYEES)),
            "X-Page-Size": str(_PAGE_SIZE),
        },
        "body": {
            "employees": page_data,
            "count": len(page_data),
            "has_more": has_more,
        },
    }

    # Add pagination token to headers (NOT to body)
    if next_token:
        response["headers"]["X-Next-Page"] = next_token

    return StringToolOutput(json.dumps(response, indent=2))


# ── Tool 4: Stripe Charges API (Cents + Header Cursor Trap) ──────────

# Simulated Stripe charges — amounts in CENTS (Stripe convention)
_STRIPE_CHARGES = [
    # Page 1 (5 charges)
    {"id": "ch_1A2B3C", "amount": 499900, "currency": "usd", "status": "succeeded",
     "description": "Enterprise Plan - Annual", "customer": "cus_acme_corp",
     "metadata": {"invoice": "INV-2024-001", "contract": "C-8801"}},
    {"id": "ch_4D5E6F", "amount": 12500, "currency": "usd", "status": "succeeded",
     "description": "API Overage - January", "customer": "cus_acme_corp",
     "metadata": {"invoice": "INV-2024-002"}},
    {"id": "ch_7G8H9I", "amount": 249900, "currency": "usd", "status": "succeeded",
     "description": "Professional Plan - Annual", "customer": "cus_globex",
     "metadata": {"invoice": "INV-2024-003", "contract": "C-8802"}},
    {"id": "ch_JKLMNO", "amount": 89900, "currency": "usd", "status": "disputed",
     "description": "Team Plan - Q1", "customer": "cus_initech",
     "metadata": {"invoice": "INV-2024-004", "dispute_reason": "product_not_received",
                   "dispute_amount": 89900, "dispute_deadline": "2024-03-15"}},
    {"id": "ch_PQRSTU", "amount": 34900, "currency": "usd", "status": "succeeded",
     "description": "Starter Plan - Monthly", "customer": "cus_umbrella",
     "metadata": {"invoice": "INV-2024-005"}},
    # Page 2 (5 charges)
    {"id": "ch_VWXY01", "amount": 499900, "currency": "usd", "status": "succeeded",
     "description": "Enterprise Plan - Annual", "customer": "cus_wayne_ent",
     "metadata": {"invoice": "INV-2024-006", "contract": "C-8803"}},
    {"id": "ch_234567", "amount": 15000, "currency": "usd", "status": "refunded",
     "description": "Support Add-on - Refunded", "customer": "cus_globex",
     "metadata": {"invoice": "INV-2024-007", "refund_reason": "duplicate_charge",
                   "original_charge": "ch_DUP001"}},
    {"id": "ch_89ABCD", "amount": 249900, "currency": "usd", "status": "succeeded",
     "description": "Professional Plan - Annual", "customer": "cus_stark_ind",
     "metadata": {"invoice": "INV-2024-008", "contract": "C-8804"}},
    {"id": "ch_EFGH01", "amount": 199900, "currency": "usd", "status": "disputed",
     "description": "Growth Plan - Annual", "customer": "cus_oscorp",
     "metadata": {"invoice": "INV-2024-009", "dispute_reason": "fraudulent",
                   "dispute_amount": 199900, "dispute_deadline": "2024-03-20"}},
    {"id": "ch_IJKL23", "amount": 34900, "currency": "usd", "status": "succeeded",
     "description": "Starter Plan - Monthly", "customer": "cus_daily_planet",
     "metadata": {"invoice": "INV-2024-010"}},
    # Page 3 (4 charges)
    {"id": "ch_MNOP45", "amount": 749900, "currency": "usd", "status": "succeeded",
     "description": "Enterprise Plus - Annual", "customer": "cus_lexcorp",
     "metadata": {"invoice": "INV-2024-011", "contract": "C-8805"}},
    {"id": "ch_QRST67", "amount": 8500, "currency": "usd", "status": "succeeded",
     "description": "API Overage - February", "customer": "cus_stark_ind",
     "metadata": {"invoice": "INV-2024-012"}},
    {"id": "ch_UVWX89", "amount": 249900, "currency": "usd", "status": "succeeded",
     "description": "Professional Plan - Annual", "customer": "cus_wayneent",
     "metadata": {"invoice": "INV-2024-013", "contract": "C-8806"}},
    {"id": "ch_YZ0123", "amount": 499900, "currency": "usd", "status": "succeeded",
     "description": "Enterprise Plan - Annual", "customer": "cus_daily_planet",
     "metadata": {"invoice": "INV-2024-014", "contract": "C-8807"}},
]

_STRIPE_PAGE_SIZE = 5
_STRIPE_CURSORS = {
    "": "cur_page2_8f3a",              # First page → cursor to page 2
    "cur_page2_8f3a": "cur_page3_c7e1",  # Page 2 → cursor to page 3
    "cur_page3_c7e1": "",                # Page 3 → no more
}

_stripe_call_count = 0


def reset_stripe_state() -> None:
    """Reset the Stripe API call counter between demo runs."""
    global _stripe_call_count
    _stripe_call_count = 0


# Pre-computed correct answers for verification
STRIPE_CORRECT_TOTAL_CHARGES = len(_STRIPE_CHARGES)
STRIPE_CORRECT_GROSS_CENTS = sum(c["amount"] for c in _STRIPE_CHARGES)
STRIPE_CORRECT_GROSS_USD = STRIPE_CORRECT_GROSS_CENTS / 100
STRIPE_CORRECT_DISPUTED_COUNT = sum(1 for c in _STRIPE_CHARGES if c["status"] == "disputed")
STRIPE_CORRECT_DISPUTED_CENTS = sum(
    c["amount"] for c in _STRIPE_CHARGES if c["status"] == "disputed"
)


@tool
def stripe_charges_api(endpoint: str) -> StringToolOutput:
    """Fetch charges from the Stripe Payments API for reconciliation.

    Base endpoint: /v1/charges
    The API returns paginated results. Pagination uses cursor-based navigation.
    Check the full response carefully for pagination information.

    IMPORTANT: All amounts are in the smallest currency unit (e.g. cents for USD).

    Args:
        endpoint: API endpoint path, e.g. "/v1/charges" or "/v1/charges?starting_after=cursor"

    Returns:
        Full HTTP response including headers and JSON body.
    """
    global _stripe_call_count
    _stripe_call_count += 1

    # Parse the endpoint
    path = endpoint.split("?")[0] if "?" in endpoint else endpoint
    params: dict[str, str] = {}
    if "?" in endpoint:
        param_str = endpoint.split("?", 1)[1]
        for pair in param_str.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                params[k] = v

    # Validate endpoint
    if path.rstrip("/") != "/v1/charges":
        return StringToolOutput(json.dumps({
            "status": 404,
            "headers": {"Content-Type": "application/json"},
            "body": {"error": {"type": "invalid_request_error",
                               "message": f"Unrecognized request URL: {path}"}},
        }, indent=2))

    # Determine cursor
    cursor = params.get("starting_after", "")

    # Common mistakes: trying page=N, offset=N, skip=N
    for bad_param in ["page", "offset", "skip", "start", "page_token"]:
        if bad_param in params:
            return StringToolOutput(json.dumps({
                "status": 400,
                "headers": {"Content-Type": "application/json"},
                "body": {"error": {"type": "invalid_request_error",
                                   "message": f"Unknown parameter '{bad_param}'. "
                                              f"Use 'starting_after' with a cursor value."}},
            }, indent=2))

    # Validate cursor
    if cursor and cursor not in _STRIPE_CURSORS:
        return StringToolOutput(json.dumps({
            "status": 400,
            "headers": {"Content-Type": "application/json"},
            "body": {"error": {"type": "invalid_request_error",
                               "message": f"Invalid cursor '{cursor}'."}},
        }, indent=2))

    # Determine slice
    if cursor == "":
        start_idx = 0
    elif cursor == "cur_page2_8f3a":
        start_idx = 5
    else:  # cur_page3_c7e1
        start_idx = 10

    page_data = _STRIPE_CHARGES[start_idx: start_idx + _STRIPE_PAGE_SIZE]
    next_cursor = _STRIPE_CURSORS[cursor]
    has_more = bool(next_cursor)

    # Build response — TRAP: pagination cursor is in HEADERS (Stripe-Cursor),
    # body only has has_more: true/false without the actual cursor value
    response = {
        "status": 200,
        "headers": {
            "Content-Type": "application/json",
            "Stripe-Version": "2024-01-01",
            "X-Request-Id": f"req_{_stripe_call_count:04d}",
        },
        "body": {
            "object": "list",
            "data": page_data,
            "has_more": has_more,
            "url": "/v1/charges",
        },
    }

    # Add cursor to HEADERS only (not in body) — this is the trap
    if next_cursor:
        response["headers"]["Stripe-Cursor"] = next_cursor

    return StringToolOutput(json.dumps(response, indent=2))
