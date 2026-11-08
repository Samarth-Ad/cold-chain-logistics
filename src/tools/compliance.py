
from langchain_core.tools import tool

from src.database.sop_retrieval import retrieve_sop_documents


@tool
def search_compliance_sop(query: str) -> str:
    """
    Search enterprise Standard Operating Procedures.

    Use this tool to retrieve:
    - Cold-chain temperature thresholds.
    - Cargo spoilage prevention procedures.
    - Port congestion and rerouting rules.
    - Shipment risk escalation procedures.

    Returns relevant SOP clauses with their source documents.
    """

    try:
        matched_docs = retrieve_sop_documents(query)

        if not matched_docs:
            return "No matching compliance clauses found."

        formatted_context = []

        for doc in matched_docs:
            source = doc.metadata.get(
                "source_file", "Unknown SOP"
            )

            file_format = doc.metadata.get(
                "file_format", "RAW"
            )

            formatted_context.append(
                f"[Source: {source} | Format: {file_format}]\n"
                f"{doc.page_content}"
            )

        return (
            "--- COMPLIANCE SOP CONTEXT ---\n"
            + "\n\n".join(formatted_context)
            + "\n------------------------------"
        )

    except Exception as exc:
        return f"Compliance SOP Retrieval Error: {exc}"
