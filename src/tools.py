"""
tools.py
--------
Day 3 deliverable, Parts 1-3: Three tools the agent can call to perform
lookups and validations that a retrieval-only pipeline can't do on its own:

    check_employee_eligibility(employee_id)
    validate_trip(employee_id, trip_type, amount, time, currency)
    calculate_reimbursement(trip_amount, policy_limit)

These read from the fictional data/employees.csv and apply the same limits
documented in the policy files (data/company_policy/), so results should
always agree with what src/rag.py would say if asked about the same limits.

Run directly for a quick manual test:
    python src/tools.py
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMPLOYEES_PATH = PROJECT_ROOT / "data" / "employees.csv"

# Standard per-trip spending limits by country, as defined in
# travel_policy_india.txt / travel_policy_us.txt / airport_policy.txt /
# expense_policy.txt / approval_policy.txt (all four agree on these numbers).
COUNTRY_POLICY_LIMITS = {
    "India": {"limit": 2000, "currency": "INR"},
    "United States": {"limit": 75, "currency": "USD"},
}

_employees_cache: pd.DataFrame | None = None


def _load_employees() -> pd.DataFrame:
    global _employees_cache
    if _employees_cache is None:
        if not EMPLOYEES_PATH.exists():
            raise FileNotFoundError(f"{EMPLOYEES_PATH} not found.")
        _employees_cache = pd.read_csv(EMPLOYEES_PATH)
    return _employees_cache


# ---------------------------------------------------------------------------
# Part 1: Employee Eligibility Tool
# ---------------------------------------------------------------------------

@dataclass
class EligibilityResult:
    employee_id: str
    found: bool
    status: str | None = None          # "Eligible" | "Not Eligible" | "Approval Required"
    country: str | None = None
    employee_type: str | None = None
    manager_approval: str | None = None
    message: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def check_employee_eligibility(employee_id: str) -> EligibilityResult:
    """
    Looks up an employee's eligibility status from data/employees.csv.

    Example:
        check_employee_eligibility("EMP001")
        -> EligibilityResult(status="Eligible", country="India", ...)

    Per the eligibility policy: an employee ID that isn't in the records is
    never assumed eligible - it's reported as not found instead.
    """
    if not employee_id or not employee_id.strip():
        return EligibilityResult(
            employee_id=employee_id or "",
            found=False,
            message="No employee ID was provided.",
        )

    employee_id = employee_id.strip().upper()
    df = _load_employees()
    match = df[df["employee_id"].str.upper() == employee_id]

    if match.empty:
        return EligibilityResult(
            employee_id=employee_id,
            found=False,
            message=f"Employee ID {employee_id} was not found in the records. "
                    f"Eligibility cannot be assumed for an unknown employee.",
        )

    row = match.iloc[0]
    return EligibilityResult(
        employee_id=employee_id,
        found=True,
        status=row["status"],
        country=row["country"],
        employee_type=row["employee_type"],
        manager_approval=row["manager_approval"],
        message=f"{employee_id} is {row['status']} ({row['country']}, {row['employee_type']}).",
    )


# ---------------------------------------------------------------------------
# Part 3: Reimbursement Calculator (defined before validate_trip, which uses it)
# ---------------------------------------------------------------------------

@dataclass
class ReimbursementResult:
    trip_amount: float
    policy_limit: float
    reimbursable_amount: float
    amount_requiring_review: float

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_reimbursement(trip_amount: float, policy_limit: float) -> ReimbursementResult:
    """
    Per expense_policy.txt: the reimbursable amount is the lower of the
    actual trip amount or the applicable country/trip spending limit.

    Example:
        calculate_reimbursement(2500, 2000)
        -> reimbursable_amount=2000, amount_requiring_review=500
    """
    trip_amount = max(0.0, float(trip_amount))
    policy_limit = max(0.0, float(policy_limit))

    reimbursable = min(trip_amount, policy_limit)
    review_amount = max(0.0, trip_amount - policy_limit)

    return ReimbursementResult(
        trip_amount=trip_amount,
        policy_limit=policy_limit,
        reimbursable_amount=reimbursable,
        amount_requiring_review=review_amount,
    )


# ---------------------------------------------------------------------------
# Part 2: Trip Validation Tool
# ---------------------------------------------------------------------------

@dataclass
class TripValidationResult:
    employee_id: str
    trip_type: str
    amount: float
    currency: str
    time: str | None
    status: str  # "Approved" | "Needs Approval" | "Not Eligible" | "Unknown Employee"
    reason: str
    reimbursement: ReimbursementResult | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def validate_trip(
    employee_id: str,
    trip_type: str,
    amount: float,
    time: str | None = None,
    currency: str | None = None,
) -> TripValidationResult:
    """
    Validates a trip request against employee eligibility and the country
    spending limit, per approval_policy.txt:
      - Unknown employee -> "Unknown Employee" (never assumed eligible)
      - "Not Eligible" status -> "Not Eligible", regardless of amount
      - "Approval Required" status -> "Needs Approval", regardless of amount
        (an employee flagged for approval needs it even under the limit)
      - "Eligible" status:
          - amount <= country limit -> "Approved"
          - amount > country limit -> "Needs Approval" (excess needs review)

    Example:
        validate_trip("EMP001", "airport", 2500, time="22:30")
        -> status="Needs Approval", reason="... exceeds the standard India
           policy limit of INR 2,000 ..."
    """
    eligibility = check_employee_eligibility(employee_id)

    if not eligibility.found:
        return TripValidationResult(
            employee_id=employee_id,
            trip_type=trip_type,
            amount=amount,
            currency=currency or "",
            time=time,
            status="Unknown Employee",
            reason=eligibility.message,
        )

    country_info = COUNTRY_POLICY_LIMITS.get(eligibility.country)
    if country_info is None:
        return TripValidationResult(
            employee_id=employee_id,
            trip_type=trip_type,
            amount=amount,
            currency=currency or "",
            time=time,
            status="Needs Approval",
            reason=f"No standard policy limit is configured for country '{eligibility.country}'; "
                   f"flagging for manual review.",
        )

    limit = country_info["limit"]
    resolved_currency = currency or country_info["currency"]
    reimbursement = calculate_reimbursement(amount, limit)

    if eligibility.status == "Not Eligible":
        return TripValidationResult(
            employee_id=employee_id,
            trip_type=trip_type,
            amount=amount,
            currency=resolved_currency,
            time=time,
            status="Not Eligible",
            reason=f"{employee_id} has eligibility status 'Not Eligible' and cannot use the "
                   f"company-sponsored travel program.",
            reimbursement=reimbursement,
        )

    if eligibility.status == "Approval Required":
        return TripValidationResult(
            employee_id=employee_id,
            trip_type=trip_type,
            amount=amount,
            currency=resolved_currency,
            time=time,
            status="Needs Approval",
            reason=f"{employee_id} has eligibility status 'Approval Required', so this trip needs "
                   f"approval regardless of the trip amount.",
            reimbursement=reimbursement,
        )

    # eligibility.status == "Eligible"
    if amount <= limit:
        return TripValidationResult(
            employee_id=employee_id,
            trip_type=trip_type,
            amount=amount,
            currency=resolved_currency,
            time=time,
            status="Approved",
            reason=f"Trip amount {resolved_currency} {amount:,.0f} is within the standard "
                   f"{eligibility.country} limit of {resolved_currency} {limit:,.0f}.",
            reimbursement=reimbursement,
        )
    else:
        return TripValidationResult(
            employee_id=employee_id,
            trip_type=trip_type,
            amount=amount,
            currency=resolved_currency,
            time=time,
            status="Needs Approval",
            reason=f"Trip amount {resolved_currency} {amount:,.0f} exceeds the standard "
                   f"{eligibility.country} policy limit of {resolved_currency} {limit:,.0f}.",
            reimbursement=reimbursement,
        )


# ---------------------------------------------------------------------------
# Tool schemas (for LLM tool/function calling in src/agent.py)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "check_employee_eligibility",
        "description": "Look up an employee's travel eligibility status, country, and employee type by employee ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string", "description": "e.g. 'EMP001'"},
            },
            "required": ["employee_id"],
        },
    },
    {
        "name": "validate_trip",
        "description": "Validate whether a trip is Approved, Needs Approval, or Not Eligible, "
                        "based on employee eligibility and the country's standard spending limit.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "trip_type": {"type": "string", "description": "e.g. 'airport', 'office', 'client_meeting'"},
                "amount": {"type": "number"},
                "time": {"type": "string", "description": "e.g. '22:30' (optional)"},
                "currency": {"type": "string", "description": "e.g. 'INR', 'USD' (optional, inferred from country if omitted)"},
            },
            "required": ["employee_id", "trip_type", "amount"],
        },
    },
    {
        "name": "calculate_reimbursement",
        "description": "Calculate the reimbursable amount and the amount requiring review, "
                        "given a trip amount and the applicable policy limit.",
        "parameters": {
            "type": "object",
            "properties": {
                "trip_amount": {"type": "number"},
                "policy_limit": {"type": "number"},
            },
            "required": ["trip_amount", "policy_limit"],
        },
    },
]


if __name__ == "__main__":
    print("=== Employee Eligibility ===")
    for emp_id in ["EMP001", "EMP002", "EMP004", "EMP999"]:
        print(check_employee_eligibility(emp_id))

    print("\n=== Trip Validation ===")
    print(validate_trip("EMP001", "airport", 1500, time="09:00"))
    print(validate_trip("EMP001", "airport", 2500, time="22:30"))
    print(validate_trip("EMP002", "airport", 500, time="09:00"))   # Approval Required regardless
    print(validate_trip("EMP004", "airport", 50, time="09:00"))    # Not Eligible
    print(validate_trip("EMP999", "airport", 100, time="09:00"))   # Unknown employee

    print("\n=== Reimbursement Calculator ===")
    print(calculate_reimbursement(2500, 2000))
    print(calculate_reimbursement(1500, 2000))