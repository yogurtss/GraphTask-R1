QUESTIONER_TOOLS = frozenset(
    {
        "inspect_entity",
        "list_relations",
        "expand",
        "execute_partial_program",
        "finalize_program",
    }
)
SOLVER_TOOLS = frozenset({"search", "text_search", "inspect_entity", "final_answer"})


def tools_for_role(role: str) -> frozenset[str]:
    if role == "questioner":
        return QUESTIONER_TOOLS
    if role == "solver":
        return SOLVER_TOOLS
    raise ValueError(f"unknown role: {role}")
