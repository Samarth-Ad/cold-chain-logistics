
from src.tools import (
    search_compliance_sop,
    query_telemetry_db,
    fetch_corridor_conditions,
)


AGENT_TOOLS = [
    query_telemetry_db,
    fetch_corridor_conditions,
    search_compliance_sop,
]


def get_agent_tools():
    """Return all tools available to the FDE agent."""

    return AGENT_TOOLS.copy()


__all__ = [
    "AGENT_TOOLS",
    "get_agent_tools",
    "query_telemetry_db",
    "fetch_corridor_conditions",
    "search_compliance_sop",
]

# Smoke test runner for local verification of all agent tools
def main():
    """Run independent smoke tests for all registered FDE agent tools."""

    print("\n" + "=" * 65)
    print("       FDE AGENT TOOLS - LOCAL VERIFICATION")
    print("=" * 65)

    test_cases = [
        {
            "name": "SQL TELEMETRY DATABASE",
            "tool": query_telemetry_db,
            "input": {
                "sql_query": """
                    SELECT TOP 2
                        Latitude,
                        Longitude,
                        Current_Temperature_C,
                        Risk_Classification
                    FROM FDE_VIEWS.VW_ACTIVE_FLEET
                """
            },
            "error_markers": ("Database Error:", "SECURITY BLOCK:"),
        },
        {
            "name": "LIVE CORRIDOR WEATHER API",
            "tool": fetch_corridor_conditions,
            "input": {
                "latitude": 33.77,
                "longitude": -118.19,
            },
            "error_markers": ("Corridor API Communication Failure:",),
        },
        {
            "name": "SOP VECTOR RETRIEVAL",
            "tool": search_compliance_sop,
            "input": {
                "query": "What are the temperature limits and mitigation steps for fresh perishables?"
            },
            "error_markers": (
                "Vector Store Retrieval Error:",
                "Error:",
            ),
        },
    ]

    results = []

    for test in test_cases:
        print(f"\n{'-' * 65}")
        print(f"TEST: {test['name']}")
        print("-" * 65)

        try:
            output = test["tool"].invoke(test["input"])

            if isinstance(output, str):
                print(output)
                failed = output.startswith(test["error_markers"])
            else:
                print(output)
                failed = False

            status = "FAILED" if failed else "PASSED"

        except Exception as exc:
            print(f"Exception: {type(exc).__name__}: {exc}")
            status = "FAILED"

        results.append((test["name"], status))
        print(f"\n[{status}] {test['name']}")

    # Final summary
    passed = sum(status == "PASSED" for _, status in results)
    failed = len(results) - passed

    print("\n" + "=" * 65)
    print("                    TEST SUMMARY")
    print("=" * 65)

    for name, status in results:
        print(f"{status:8} | {name}")

    print("-" * 65)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("=" * 65)

    if failed == 0:
        print("\nAll agent tools are responding successfully!")
    else:
        print("\nSome tools failed. Check their output above.")


if __name__ == "__main__":
    main()
