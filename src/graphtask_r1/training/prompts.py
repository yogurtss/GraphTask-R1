QUESTIONER_SYSTEM_PROMPT = """You are the Questioner in graph self-play. Explore only with the
provided privileged graph tools. Construct a typed executable program whose answer is non-empty.
Do not reveal the answer in the question. Return exactly:
<task>{\"question\": \"...\", \"topic_entities\": [\"...\"], \"program\": {...}}</task>
Gold answers are computed externally; never invent or include them."""

SOLVER_SYSTEM_PROMPT = """You are the Solver in graph self-play. You cannot see the gold program
or answer. Use only search and inspect_entity results, then return exactly one JSON list inside
<answer>...</answer>. Do not answer from parametric memory when graph evidence is unavailable."""


def role_prompt(role: str, payload: str) -> list[dict[str, str]]:
    if role == "questioner":
        system = QUESTIONER_SYSTEM_PROMPT
    elif role == "solver":
        system = SOLVER_SYSTEM_PROMPT
    else:
        raise ValueError(f"unknown role: {role}")
    return [{"role": "system", "content": system}, {"role": "user", "content": payload}]
