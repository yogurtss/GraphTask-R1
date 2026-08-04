from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from graphtask_r1.schema import Triple


class GraphOverlay(BaseModel):
    model_config = ConfigDict(frozen=True)

    added: tuple[Triple, ...] = ()
    removed: tuple[Triple, ...] = ()

    def apply(self, triples: tuple[Triple, ...]) -> tuple[Triple, ...]:
        removed = set(self.removed)
        result = (set(triples) - removed) | set(self.added)
        return tuple(sorted(result, key=Triple.sort_key))
