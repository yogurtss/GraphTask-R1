# Four Invention Points: Slide-Ready English Copy

## Slide 1 — A DSL-Based Interaction Standard for LLM–Graph Communication

**Subtitle**

A typed, validated, and replayable contract between an LLM agent and a graph environment.

**Core mechanism**

- The LLM first discovers the graph environment's supported capabilities and operation schemas.
- Every request is converted into a structured DSL action with explicit types, parameters, constraints, and expected outputs.
- The graph environment validates the action before execution and returns structured observations, results, or rejection codes.
- A trace ID links the request, action, environment state, execution trace, and result for deterministic replay and audit.

**Key technical details**

- A formal grammar and schema define valid graph operations and message types.
- Requests, actions, observations, and results are separated into explicit protocol objects.
- Environment state and interaction records remain JSON-serializable and replayable.
- Backend-specific graph operations are hidden behind a backend-neutral interaction contract.
- Timeouts, retries, caching, and trace IDs can be enforced at the protocol boundary.

**Inventive contribution**

- Replaces unconstrained natural-language tool use with a machine-checkable interaction standard.
- Prevents interface ambiguity by separating intent, execution, observation, and result.
- Enables the same LLM policy to operate across heterogeneous graph backends without changing the interaction semantics.
- Makes graph-agent behavior testable, auditable, and reproducible at the protocol level.

**Suggested callout**

> The DSL is not merely a command syntax; it is the executable contract that governs the entire LLM–graph interaction lifecycle.

**Figure caption**

The DSL contract mediates all communication between the LLM agent and the graph environment, enforcing typed messages, schema validation, capability discovery, structured errors, and end-to-end traceability.

**Speaker notes**

This invention introduces a formal interaction layer between an LLM and a graph environment. Instead of allowing the model to communicate through unconstrained natural language or backend-specific APIs, all exchanges are represented as typed DSL objects. The contract defines what the agent may request, how the environment validates and executes an action, and how observations and results are returned. Because each interaction is associated with a trace ID and a serializable state, the complete process can be audited and replayed. The result is a backend-neutral standard that improves interoperability, safety, and reproducibility.

---

## Slide 2 — Questioner–Solver Co-Evolution with Self-Generated Graph Data

**Subtitle**

Two specialized agents continuously generate, solve, verify, and learn from graph-native tasks.

**Core mechanism**

- The Questioner explores a graph snapshot and generates a question together with an executable candidate program.
- The Solver receives the question independently and produces a predicted answer or solution program.
- The graph executor runs the candidate program and returns an execution-grounded answer and structured feedback.
- Accepted tasks, programs, traces, answers, difficulty scores, and rejection reasons are stored in a data archive.
- The Questioner learns to generate more valuable and challenging tasks, while the Solver learns to solve them more accurately.

**Optimization loop**

1. **Generate:** sample a graph state and synthesize a candidate task.
2. **Solve:** obtain an independent prediction from the Solver.
3. **Execute:** run the associated program in the graph environment.
4. **Verify:** evaluate validity, correctness, informativeness, and difficulty.
5. **Archive:** retain certified training examples and structured rejection records.
6. **Update:** optimize both agents using role-specific feedback.

**Inventive contribution**

- Creates training data directly from graph structure and executable graph operations, reducing dependence on manually labeled datasets.
- Uses asymmetric but coupled Questioner and Solver objectives, allowing each agent to provide a moving curriculum for the other.
- Grounds the self-play loop in graph execution rather than mutual agreement between language models.
- Preserves both accepted examples and structured failure signals, improving sample efficiency and controllability.

**Suggested callout**

> The system does not merely self-play; it self-generates executable supervision from the graph and co-evolves both data generation and problem solving.

**Figure caption**

The Questioner synthesizes graph-grounded tasks and programs, the Solver produces independent predictions, and graph execution supplies verified feedback that updates both agents and enriches the training archive.

**Speaker notes**

The second invention is a dual-agent optimization framework. The Questioner is responsible for discovering useful supervision within the graph, while the Solver is responsible for answering the generated questions. Their roles are coupled but deliberately asymmetric. The Questioner is rewarded for validity, informativeness, novelty, and appropriate difficulty; the Solver is rewarded for verified correctness. Crucially, neither agent decides the ground truth. Execution in the graph environment supplies the verification signal. This design turns unlabeled graph structure into an expanding training corpus and enables continuous co-evolution without losing control over data quality.

---

## Slide 3 — Curriculum-Based Multi-Stage Reward Mechanism

**Subtitle**

Progressively increase task difficulty only after validity, executability, and correctness have been established.

**Four-stage curriculum**

- **Stage 1 — Syntax validity:** reward well-formed DSL programs and valid schemas.
- **Stage 2 — Executability:** reward programs that complete within bounded resources on the selected graph state.
- **Stage 3 — Verified correctness:** reward unambiguous questions whose answers are deterministically supported by execution.
- **Stage 4 — Difficulty and novelty:** reward tasks that challenge the current Solver and add non-redundant coverage to the archive.

**Adaptive control**

- The curriculum controller monitors stage-specific pass rates and Solver competence.
- Difficulty thresholds increase when the Solver becomes reliable at the current level.
- Thresholds decrease or the system returns to earlier stages when validity or precision deteriorates.
- Rejected candidates receive structured feedback and are regenerated rather than discarded as undifferentiated failures.

**Reward decomposition**

`R = w_syn R_syn + w_exec R_exec + w_corr R_corr + w_diff R_diff + w_nov R_nov`

The weights and acceptance thresholds are stage-dependent, allowing the optimization target to shift from feasibility to correctness and finally to difficulty.

**Inventive contribution**

- Separates quality requirements into ordered gates instead of optimizing a single entangled reward from the beginning.
- Preserves precision by making difficulty optimization conditional on prior validity and correctness.
- Automatically adjusts data complexity using both generator pass rates and Solver competence.
- Converts rejection reasons into targeted learning signals for regeneration and policy improvement.

**Suggested callout**

> Difficulty is earned, not assumed: the system advances only when earlier quality gates remain stable.

**Figure caption**

Candidate tasks pass through syntax, execution, correctness, and difficulty gates, while an adaptive curriculum controller changes thresholds using pass rates and Solver competence.

**Speaker notes**

The third invention addresses a common failure mode in self-generated training data: pushing for difficult tasks too early often reduces validity and precision. Our reward mechanism therefore decomposes quality into four ordered stages. Early optimization focuses on syntax and executability, followed by deterministic correctness, and only then emphasizes difficulty and novelty. An adaptive controller adjusts thresholds based on observed pass rates and Solver competence. This produces an automatic curriculum in which task complexity rises with system capability, while structured feedback routes failed candidates back to the appropriate stage for targeted regeneration.

---

## Slide 4 — Execution-Driven Certification for Hallucination-Resistant Ground Truth

**Subtitle**

Gold answers are derived exclusively from certified program execution, never from free-form LLM output.

**Certification pipeline**

1. **LLM proposal:** generate a candidate question and its associated DSL program.
2. **DSL validation:** check syntax, schema, types, permissions, and resource declarations.
3. **Bounded execution:** execute against a specified graph snapshot with an explicit seed and bounded limits.
4. **Trace and policy checks:** verify determinism, access policies, invariants, and result integrity.
5. **Program certification:** issue a certificate only when all validation and execution gates pass.
6. **Gold derivation:** compute the reference answer solely from the certified execution result.
7. **Replayable archival:** store the question, program, graph snapshot, seed, trace, result, and answer as one provenance-linked record.

**Failure handling**

- Failed candidates never enter the gold-answer dataset.
- Each failure retains a structured rejection reason, such as a syntax error, schema mismatch, policy violation, resource limit, non-determinism, or ambiguous result.
- Rejection records are routed back to the generator for targeted regeneration and policy improvement.

**Inventive contribution**

- Establishes an unbroken provenance chain from question to certified program, deterministic trace, and gold answer.
- Eliminates LLM-generated ground truth as a source of label hallucination.
- Converts open-ended self-play into a bounded, auditable, and reproducible closed-loop system.
- Supports exact replay by preserving the graph snapshot, seed, execution policy, program, and trace.
- Makes acceptance and rejection behavior independently testable and suitable for regulated or high-assurance settings.

**Suggested callout**

> A reference answer is trusted because its generating program was certified and executed—not because an LLM stated it confidently.

**Figure caption**

Only programs that pass DSL validation, bounded graph execution, and deterministic trace checks may generate gold answers; all accepted artifacts are preserved in a replayable archive.

**Speaker notes**

The fourth invention is the trust anchor of the entire system. A language model may propose a question and a candidate program, but it is never permitted to declare the reference answer. The program must first pass formal validation, bounded execution, and trace-level policy checks. Only the output of a certified execution is transformed into the gold answer. The system then archives every artifact required for replay, including the graph snapshot, seed, program, and trace. This architecture prevents hallucinated labels by construction and turns self-play into a controlled and reproducible data-generation pipeline.

---

## Optional Overview Slide — One Integrated Closed-Loop System

**Title**

From Standardized Interaction to Certified Self-Evolution

**On-slide summary**

- **Interaction standard:** a typed DSL connects LLM agents to graph environments.
- **Self-evolution:** Questioner and Solver co-evolve using graph-native, self-generated tasks.
- **Adaptive curriculum:** staged rewards increase difficulty without sacrificing precision.
- **Execution certification:** only certified program execution can produce gold answers.

**Integrated value proposition**

Together, the four inventions transform open-ended LLM self-play into a standardized, self-improving, precision-controlled, and fully replayable graph-learning system.

