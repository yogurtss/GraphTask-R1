# Contributor instructions

- Keep graph backends, RNGs, configs, policies, and archives instance-scoped.
- Pass seeds explicitly; never depend on module-global random state.
- Derive gold answers only by executing the certified program.
- Preserve structured rejection reasons and reward components.
- Keep environment state JSON-serializable and replayable.
- External graph calls must use timeouts, retries, caching, and trace IDs.
- Add or update unit and integration tests for every behavior change.
- Run `make lint`, `make typecheck`, and `make test` before publishing.
- Use ToyGraph and bounded limits before external data or GPU runs.

