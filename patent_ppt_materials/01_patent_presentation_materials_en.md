# GraphTask-R1 Patent Presentation Materials

## Recommended Patent Title

### Recommended concise title

**DSL-Driven Self-Evolving Graph Agent**

### Recommended formal title

**A Method for DSL-Driven Self-Evolution of a Graph Agent**

This title retains the two most distinctive concepts—an executable domain-specific language and self-evolution—without listing every implementation detail. In the specification, a **graph agent** may be defined as a computer-implemented agent that converts a natural-language graph task into a constrained executable graph program, invokes a graph execution environment, and produces a verifiable graph-domain output.

### Alternative titles

1. **A DSL-Based Self-Evolving Agent for Graph Task Generation** — emphasizes automatic graph-task generation.
2. **A Self-Evolving Graph Agent with Executable Task Certification** — emphasizes verifiability.
3. **A Method for Self-Evolving Graph Task Generation and Management** — broadly covers the task lifecycle.
4. **Executable-DSL Self-Evolution for Graph Agents** — short and presentation-friendly.

### One-sentence invention concept

**A graph agent jointly generates a natural-language task and a bounded typed graph program, derives the reference answer only through certified program execution, and improves task generation and task solving through an interface-aware, replayable self-evolution loop.**

---

# Part I. Background of the Invention

## Page 1 — Technical Field of the Invention

### Suggested slide headline

**From Graph Reasoning to Verifiable Self-Evolving Graph Agents**

### Extended background

Graphs are a fundamental representation for data whose meaning depends on relationships. A graph may encode entities, directed relations, attributes, qualifiers, temporal conditions, and multi-hop dependencies. Graph-structured data is widely used in enterprise knowledge bases, financial relationship analysis, recommendation systems, biomedical discovery, cybersecurity, industrial maintenance, and data-lineage management.

Graph reasoning differs from ordinary text generation. A system must not only understand the semantics of a user request; it must also preserve relation direction, entity identity, variable dependencies, filters, qualifiers, set operations, and execution order. A plausible natural-language answer is insufficient when the requested result must be proven to follow from a particular graph snapshot.

Traditional graph databases provide precise query languages such as SPARQL and Cypher. These languages are executable and auditable, but they normally require trained users to understand graph schemas and manually construct queries. Large language models reduce this interaction barrier by allowing a user to express a graph task in natural language. This development has led from direct prompting, to language-model-generated queries, to agents that plan and invoke graph tools over multiple steps.

The next technical challenge is not merely to let an agent access graph data. It is to establish a stable interface through which the agent can generate graph operations, execute them within controlled limits, verify the resulting answer, learn from distinct failure modes, and continuously create reliable training tasks. The present invention addresses this challenge by combining a typed executable DSL, certified task generation, conditional curriculum rewards, deterministic task management, and separate Questioner/Solver updates.

### Technology evolution

**Manual graph query → Prompt-based graph QA → Tool-using graph agent → Executable-DSL graph agent → Verifiable self-evolving graph agent**

| Technology stage | Typical mechanism | Main capability introduced | Direction of further development |
|---|---|---|---|
| Manual graph query | A specialist writes SPARQL, Cypher, or another query | Precise and deterministic graph access | Natural-language accessibility |
| Prompt-based graph QA | Graph triples or retrieved subgraphs are inserted into an LLM prompt | Low-friction natural-language interaction | Stronger structural grounding and execution constraints |
| Query-generation system | An LLM generates a formal graph query and sends it to a database | Executable graph retrieval | Safer program space, richer reasoning, and consistent validation |
| Tool-using graph agent | The model plans, calls graph tools, observes results, and continues | Dynamic interaction with external graph data | Unified state, bounded control flow, and replayability |
| Executable-DSL graph agent | The model emits a typed program under an explicit schema and budget | A common generation, execution, and verification contract | Certified task generation and adaptive training |
| Self-evolving graph agent | The system generates, certifies, selects, archives, and learns from new tasks | A closed data–verification–training loop | Stable long-term evolution with auditable state |

### Convergence of current technical directions

The invention is positioned at the intersection of four active technical directions. This positioning describes the technical context; it is not a claim of being the first or only implementation.

| Technical direction | General objective | Corresponding mechanism in the invention |
|---|---|---|
| Graph + LLM grounding | Connect language understanding to structured external knowledge | An instance-scoped graph snapshot and relation catalogue serve as the executable knowledge environment |
| Agentic AI | Allow a model to plan and perform multi-step tasks | Questioner and Solver roles generate and solve graph tasks under explicit contracts |
| DSL and program synthesis | Convert language intent into structured machine-executable operations | A bounded, typed GraphScript or functionally equivalent DSL defines the action space |
| Verifiable self-evolution | Improve an agent using automatically generated tasks and reliable feedback | Execution produces gold answers and certificates; curriculum rewards and archive admission control learning |

### Why this technical field matters

- Graphs provide structured, updateable, and organization-specific knowledge that may not be contained in model parameters.
- Language models provide flexible intent understanding but require an external mechanism to enforce exact graph semantics.
- Executable programs provide deterministic operations, explicit intermediate state, and inspectable failure points.
- Certified self-generated tasks can expand training data without treating an unverified model answer as ground truth.
- Replayable state, traces, and reason codes support diagnosis, governance, and later technical inspection.

### Slide-ready text

- Graph data captures entities, relations, attributes, qualifiers, and multi-hop dependencies.
- LLMs make graph systems accessible through natural language, but free-form generation cannot guarantee graph-faithful answers.
- A typed executable DSL converts language intent into bounded, verifiable graph operations.
- Certified task generation and role-separated learning move a Graph Agent from static capability toward controlled self-evolution.

### Optional speaker notes

“Graph reasoning is moving from manually written queries toward autonomous agents. The key frontier is no longer simple graph access. A practical graph agent must connect natural language, executable structure, verifiable results, and continuous learning. Our invention places a typed DSL at the center of this connection and uses certified execution to close the self-evolution loop.”

### Formal field statement

The invention relates to artificial intelligence, graph-data processing, computer-implemented agents, program synthesis, domain-specific languages, and reinforcement learning. More particularly, it relates to a method in which a constrained executable graph program connects a language model to a graph execution environment, and in which certified task generation, staged reward computation, deterministic task management, and role-separated updates are used to train a self-evolving graph agent.

### Key takeaway

**The invention advances a Graph Agent from merely accessing a graph to generating, executing, certifying, managing, and learning from graph tasks.**

---

## Page 2 — Description of the Prior Art

### Suggested slide headline

**Existing Graph-Agent Paradigms Solve Different Parts of the Problem**

### Overview

Existing LLM-based graph systems generally follow one or more of four paradigms:

1. **Prompt-based graph augmentation.** Graph triples, paths, descriptions, or retrieved subgraphs are converted into text and inserted into the prompt. The LLM then produces a free-form answer.
2. **Tool-based graph agents.** The model alternates between planning and invoking tools such as graph search, entity inspection, relation traversal, or text retrieval.
3. **Query-language or code generation.** The model generates SPARQL, Cypher, Python, or another executable representation that is evaluated by an external engine.
4. **Questioner–Solver self-play.** A task generator creates new questions, a solver attempts them, and the resulting score is used to update one or both roles.

These paradigms are valuable and complementary. Prompting simplifies natural-language access; tool use permits dynamic retrieval; executable queries improve precision; and self-play reduces dependence on a fixed human-labelled dataset. However, these functions are often implemented through separate interfaces, reward definitions, and state-management mechanisms.

### Prior-art comparison

| Paradigm | Typical pipeline | Main output | Strength | Limitation relevant to this invention |
|---|---|---|---|---|
| Prompt-based graph QA | Question + textualized triples/subgraph → LLM → answer | Free-form answer | Simple integration with general LLMs | Graph structure is compressed into context and the answer is not forced through execution |
| Retrieval-augmented graph QA | Question → retrieve graph evidence → add evidence to prompt → answer | Answer + retrieved context | Scales beyond a small graph context | Retrieval and final answer generation may use different validation rules |
| Tool-based graph agent | Question → plan → tool call → observation → repeat → answer | Variable-length tool trace | Can access dynamic or private graphs | Cost and error propagation grow with turns; traces may be framework-specific |
| Query-language generation | Question → SPARQL/Cypher/code → execute → result | Query + database result | Precise execution and database compatibility | Unrestricted query/code spaces may be difficult to control uniformly |
| Self-play task generation | Proposer generates task → Solver attempts → reward → update | New tasks and updated model | Expands the training distribution | Unanswerable tasks and interface failures may contaminate learning |
| Persistent-memory skill evolution | Agents generate or retain tasks or skills over rounds | Evolving memory or task pool | Supports long-term adaptation | Memory admission may not be tied to program certification and graph evidence |

### Relationship among the paradigms

The paradigms may be viewed as successive additions of capability:

- Prompting adds natural-language accessibility.
- Retrieval and tools add external knowledge access.
- Query generation adds executability.
- Self-play adds adaptive data generation.
- Persistent memory adds multi-round retention.

The present invention does not rely on any one capability in isolation. It focuses on the cooperating technical chain that joins them through a common executable contract.

### Slide-ready text

- Prompt methods place graph evidence in text and ask the LLM to answer.
- Tool agents repeatedly plan, invoke graph tools, and consume observations.
- Query-generation methods produce executable database instructions.
- Self-play methods generate new tasks and use Solver performance as feedback.
- Existing capabilities are usually separated; generation, execution, certification, reward, archive, and update do not necessarily share one contract.

### Optional speaker notes

“Prior work has shown that language models can use graph context, call tools, generate queries, and create tasks. Our distinction should not be framed as any one of these broad ideas. The relevant gap is the absence of a unified and replayable contract spanning task generation, bounded execution, gold derivation, difficulty measurement, archive admission, and role-separated learning.”

---

## Page 3 — Problems of the Prior Art

### Suggested slide headline

**Graph Access Is Not the Same as Verifiable Graph Intelligence**

### Detailed problem matrix

| Technical problem | Prompt-based manifestation | Tool-based or self-play manifestation | Technical consequence |
|---|---|---|---|
| Loss of graph structure | Triples and paths are flattened into text; direction and qualifiers may be diluted | Each tool call exposes only a partial local observation | The answer may be plausible but inconsistent with the actual graph |
| No mandatory execution path | The model may answer from parametric memory | The model may ignore or misread tool observations | It is difficult to prove that the answer came from the designated graph |
| Unbounded or unstable control flow | Behavior varies with prompt length and context | The agent may issue many invalid or repetitive calls | Latency, cost, and resource use are difficult to predict |
| Fragmented intermediate state | Free-form reasoning is not machine-executable | Tool traces depend on framework-specific messages | Runs are difficult to replay, compare, migrate, or audit |
| Ambiguous failure attribution | A wrong answer becomes one undifferentiated failure | Parse, execution, and semantic failures may receive the same score | The signal does not identify which capability should improve |
| Distorted self-play difficulty | A task appears hard because the Solver cannot use the interface | Low Solver success is treated as desirable frontier difficulty | The Questioner may be rewarded for malformed tasks |
| Unreliable generated gold | The generator may include a guessed answer | A weak Solver output may be reused as a pseudo-label | Incorrect labels enter the self-evolution loop |
| Poor task-library control | Tasks lack a program-level identity | Duplicate, trivial, or unanswerable tasks accumulate | Archive growth does not imply useful coverage |
| Role interference | One adapter or mixed update changes both roles | The difficulty judge moves during evaluation | Difficulty measurements become unstable |
| Weak reproducibility | Random, graph, and policy state may be implicit | The same round may produce a different dataset | Training cannot be reliably resumed or investigated |

### Root causes

1. **No unified program contract.** Natural-language generation, tool execution, dataset representation, and evaluation use different interfaces.
2. **No execution-grounded certification boundary.** Generated tasks are not consistently accepted according to the same graph snapshot, schema, relation catalogue, and budget.
3. **No separation between interface readiness and semantic difficulty.** Self-play feedback confuses “the Solver could not use the interface” with “the task is semantically challenging.”

### Technical problem to be solved

**How can a language-model-based graph agent automatically generate and solve new graph tasks while ensuring that each accepted task is executable, its reference answer is derived from the designated graph environment, the training signal distinguishes interface failure from semantic difficulty, and the evolving task distribution remains bounded, replayable, and auditable?**

### Desired technical properties

| Property | Meaning in the proposed system |
|---|---|
| Executable | A model output can be parsed and run by a graph executor |
| Bounded | Operators, relation access, edge visits, and returned entities are limited |
| Verifiable | The answer and evidence originate from certified execution |
| Diagnosable | Failures are preserved as structured reasons and reward components |
| Replayable | Graph snapshot, seed, policy, data, and adapter state can be restored |
| Evolvable | Newly certified tasks improve the Solver and guide the next Questioner |
| Detectable | Core artifacts can be observed through programs, traces, errors, or certificates |

### Slide-ready text

- Textual graph prompting weakens structural guarantees.
- Multi-turn tools introduce variable control flow, latency, and framework-specific traces.
- Low Solver performance may indicate interface failure rather than true semantic difficulty.
- Unverified pseudo-labels and uncontrolled archives can destabilize self-evolution.
- The missing element is one contract across **generation → execution → certification → reward → archive → training**.

### Optional speaker notes

“The central problem is not whether an LLM can call a graph. The central problem is whether every accepted task and answer can be tied to bounded execution under a known graph state, and whether that evidence can drive stable self-evolution.”

### Recommended visual

Use **figures/fig1_related_work_comparison.png**.
+

---

# Part II. Summary of the Invention and Core Invention Points

## Page 4 — Summary of the Invention

### Suggested slide headline

**A Certified Executable-Graph Loop for Self-Evolving Graph Agents**

### Patent-style summary

The invention provides a computer-implemented method for training a self-evolving graph agent. The method obtains an instance-scoped graph snapshot, a relation catalogue, seed entities, base tasks, explicit random seeds, and execution budgets. A Questioner model generates, in a single output, a natural-language question and a corresponding bounded typed graph program. A parser and graph executor validate and execute the program under schema, relation, handle, operation-count, edge-visit, and returned-entity constraints. A task is certified only when the question–program pair satisfies predetermined checks, and the reference answer is derived from execution of the certified program rather than generated by the model.

A frozen Solver is evaluated on certified tasks. Parsing success, conditional execution success, and conditional semantic success are measured separately. A staged reward first promotes production of a complete question–program contract, then graph grounding and executability, and finally tasks near a target semantic frontier. The semantic-frontier contribution is gated by interface readiness so that malformed or unexecutable outputs are not mistaken for difficult tasks.

Certified tasks are deterministically admitted to a persistent archive according to a Solver pass-rate window, structural novelty, textual novelty, and structured rejection reasons. Within the same training round, the Solver dataset is rebuilt from base tasks, historical archived tasks, and newly admitted tasks. Questioner and Solver use separate adapters and are updated in a defined order. The round state, dataset hashes, adapter identifiers, task certificates, and execution traces are stored for replay and audit.

### System objective

The method transforms open-ended model self-play into a controlled process in which:

- the action space is executable and bounded;
- the reference answer has a traceable source;
- interface competence is separated from semantic competence;
- task admission is deterministic and explainable;
- Questioner and Solver learning remain role-separated; and
- every round produces a recoverable technical state.

### Inputs and outputs

| Category | Examples |
|---|---|
| Instance-scoped inputs | Graph snapshot ID, graph-backend instance, relation catalogue, seed entities, base tasks |
| Control inputs | GraphScript profile, allowed operators, maximum follow limit, edge-visit budget, return budget |
| Learning inputs | Questioner adapter, Solver adapter, curriculum stage, target frontier, archive policy |
| Intermediate artifacts | Question, graph program, parse result, execution trace, candidate task, reward breakdown |
| Certified artifacts | Execution-derived gold, witness facts, canonical program signature, task certificate |
| Training outputs | Updated Questioner adapter, updated Solver adapter, rebuilt Solver dataset |
| System outputs | Verified graph-QA model, controlled task archive, replayable manifests and traces |

### Functional modules

| Module | Input | Core operation | Output |
|---|---|---|---|
| Instance-scoped environment | Graph snapshot, catalogue, seeds, budgets | Fix graph state and allowed execution boundary | Replayable environment state |
| Questioner | Seed context, relations, task policy | Jointly generate question and graph program | Candidate {question, program} |
| DSL parser and validator | Generated program | Exact-JSON, schema, operator, handle, and relation checks | Valid program or structured error |
| Bounded executor | Valid program and graph backend | Execute under relation and resource constraints | Answer, evidence, budget use, trace |
| Task certification | Question, program, execution result | Check executability, alignment, leakage, non-empty result, consistency | Certificate or rejection reasons |
| Frozen Solver evaluation | Certified question with hidden gold | Measure parse, conditional execution, and conditional semantic success | Interface and difficulty signals |
| Curriculum reward | Generation milestones and Solver signals | Compute production, grounding, and frontier contributions | Dense reward breakdown |
| Deterministic archive | Certificate and Solver statistics | Apply difficulty and novelty policies in stable order | Accepted task or rejection |
| Role-separated updater | Role datasets and adapters | Update roles in a defined sequence | New role-specific adapters |
| Round-state manager | Config, hashes, adapters, decisions | Serialize and validate resumable state | Plan, manifest, metrics, audit artifacts |

### Ordered method flow

| Step | Operation | Technical result |
|---|---|---|
| S1 | Obtain graph snapshot, relation catalogue, seeds, base tasks, explicit random seed, and budgets | A bounded, replayable episode environment |
| S2 | Generate a natural-language question and typed graph program together | An explicit language-to-program task pair |
| S3 | Parse, validate, and execute the program; derive gold only from certified execution | A verified answer, evidence trace, and task certificate |
| S4 | Evaluate with a frozen Solver and compute separated interface/semantic signals | A meaningful task-difficulty estimate |
| S5 | Deterministically admit tasks by difficulty and novelty | A controlled persistent archive |
| S6 | Rebuild the Solver dataset in the same round from base, historical, and new tasks | A curriculum-controlled training dataset |
| S7 | Update separate Questioner/Solver adapters and store round state | A replayable next-round self-evolution state |

### Closed-loop summary

**Instance-scoped graph environment → Questioner joint generation → DSL validation and bounded execution → certification → frozen-Solver evaluation → deterministic archive admission → same-round Solver learning → next-round self-evolution**

### Technical effects

- Converts language generation into a machine-executable and inspectable graph-operation process.
- Ensures gold, evidence, and traces are tied to a designated graph snapshot.
- Prevents interface failures from being misclassified as useful semantic difficulty.
- Controls duplicate tasks, difficulty drift, unanswerable samples, and archive contamination.
- Prevents Questioner and Solver parameters from overwriting one another.
- Produces JSON-serializable, resumable, and auditable training state.
- Creates observable technical artifacts that support practical detectability.

### Slide-ready text

- **Generate:** output a natural-language question and typed GraphScript together.
- **Certify:** validate the program and derive gold through bounded execution.
- **Evaluate:** use a frozen Solver to separate interface readiness from semantic success.
- **Manage:** control archive admission using difficulty and novelty.
- **Evolve:** update separate Questioner and Solver adapters in a defined same-round loop.

### Optional speaker notes

“Our system turns self-play into a certified data pipeline. The Questioner supplies both a question and an executable specification. The graph executor is the ground-truth authority. Solver feedback is decomposed so that malformed programs are not treated as difficult tasks. Accepted tasks enter a controlled archive and can be consumed by the Solver in the same round.”

### Recommended architecture figure

Use **figures/fig2_dsl_adversarial_coevolution_gpt_image_v2.png**. The editable vector fallback is **figures/fig2_curriculum_v3_architecture_dsl_coevolution.svg**.

The revised GPT Image figure intentionally uses a single left-to-right spine: answer-free context → Questioner-emitted graph DSL task contract → certified execution → conditional Questioner–Solver duel. Small graph, code, certification, archive and update motifs add visual explanation without restoring the crossing arrows or dense lane structure of the earlier draft.

---

## Page 5 — Invention Point 1: DSL-Driven Bounded Executable Interface

### Core proposition

A common, constrained, typed, and executable DSL is placed between the language model and the graph environment. Questioner, Solver, training data, reward functions, and evaluation share this program contract.

### Why a DSL is used

A general-purpose language can express arbitrary operations, creating safety and validation difficulties. A collection of ad hoc tools provides access but may fragment the action space into framework-specific calls. A graph DSL narrows the action space to operations that are meaningful for graph reasoning and can be checked before execution.

### Main technical components

| Component | Example implementation | Technical role |
|---|---|---|
| Versioned program object | GraphScript version and operation list | Makes parser and executor behavior explicit |
| Typed operation schema | Resolve, follow, filter, intersect, count, query, verify, select, emit | Prevents arbitrary operations |
| SSA-style handles | h0 to h63, with each output assigned once | Makes intermediate dependencies explicit |
| Relation/qualifier catalogue | Episode-specific whitelist | Prevents access outside the graph contract |
| Explicit direction and limits | In/out direction and follow limit | Preserves semantics and bounds traversal |
| Terminal output contract | A unique final emit operation | Identifies the result without text ambiguity |
| Execution budgets | Operation, edge-visit, and returned-entity limits | Controls cost and graph expansion |
| Structured trace | Operators, handles, evidence additions, usage, and errors | Supports replay, diagnosis, and audit |

### Representative operation groups

| Group | Representative operations | Purpose |
|---|---|---|
| Entry and resolution | all_entities, resolve_entity | Establish a starting entity set |
| Traversal | follow | Traverse a relation in a specified direction |
| Set operations | intersect, union | Combine intermediate entity sets |
| Filtering | filter_type, filter_literal, filter_qualifier | Enforce type, attribute, or qualifier conditions |
| Aggregation/comparison | count, verify, select_between, select_among | Produce numeric, Boolean, or comparative outputs |
| Attribute/relation query | query_attribute, query_relation, qualifier-aware queries | Extract structured values |
| Output | emit | Designate the final answer handle |

These operators are an implementation example. Protection should describe functional operator classes and execution constraints rather than require a specific product name or DSL version.

### Validation hierarchy

1. The output contains exactly one machine-readable program object.
2. The program version and schema are supported.
3. Every operator belongs to the permitted profile.
4. Every input handle refers to a valid earlier output.
5. Relations and qualifiers belong to the episode catalogue.
6. Traversal and materialization remain within configured budgets.
7. The final operation identifies one program output.
8. Execution produces a traceable answer or structured rejection reason.

### Difference from ordinary tool calling

| Ordinary tool-use pattern | Proposed executable-DSL interface |
|---|---|
| Multiple conversational turns | One complete program can be generated before execution |
| Tool semantics distributed across prompts and wrappers | Operations share one typed schema |
| Variable framework-specific trace | Stable program object and step trace |
| Errors may surface only after several calls | Schema, handle, relation, and budget errors are detected early |
| Final answer may be generated separately from tool results | Concrete answer is produced through execution |
| Difficult to reuse trajectories across frameworks | The same parser, schema, operators, and executor are reused |

### Technical effect chain

**Typed DSL + relation whitelist + bounded execution + structured trace → controlled graph access + deterministic intermediate state + diagnosable failure + replayable answer production**

### Protection focus

The broad concept should cover:

- a graph-domain executable program generated from a natural-language task;
- a restricted set of typed graph operators;
- validation against an episode-specific graph contract;
- bounded graph execution; and
- a concrete graph result and execution trace.

The principal claim should avoid depending solely on “GraphScript,” version 0.3, a maximum of 64 operations, or one exact operator list. Those details are better embodiments or fallback limitations.

### Slide-ready text

- One typed contract is shared by generation, execution, training, and evaluation.
- Operators, handles, relations, directions, and final output are explicit.
- Execution is bounded by operation, traversal, edge-visit, and return limits.
- Every failure maps to a program stage and structured reason.
- The result is a verifiable and replayable interface, not a free-form reasoning trace.

### Optional speaker notes

“The DSL is not merely an output format. It is the technical boundary of the system. It defines what the agent may do, how intermediate state is represented, how graph access is limited, and how the result is replayed. This shared contract makes later certification and curriculum mechanisms possible.”

---

## Page 6 — Invention Point 2: Joint Task Generation and Execution Certification

### Core proposition

The Questioner jointly generates a natural-language question and its corresponding graph program. The program is executed under a defined graph state, and the actual execution result—rather than any answer generated by the Questioner—is used as the reference answer.

### Why joint generation matters

Generating only a question does not provide a machine-checkable definition of its meaning. Generating only a program does not provide a natural-language task for training. Producing the pair creates an explicit alignment object:

**Natural-language intent ↔ executable graph semantics**

The pair can be tested for structural validity, executability, consistency, and answer leakage before becoming training data.

### Certification pipeline

| Layer | Example check | Failure prevented |
|---|---|---|
| Output contract | Question and program are both present | Partial proposals |
| Machine format | Exact JSON and schema validity | Unparseable model text |
| Program structure | Operators, handles, types, and output are valid | Broken execution graphs |
| Episode grounding | Seeds and relations match the allowed context | Ungrounded graph access |
| Resource safety | Operation and graph-access budgets are respected | Runaway traversal |
| Executability | Program completes successfully | Tasks with no operational meaning |
| Answer validity | Result is non-empty and has acceptable cardinality | Trivial or unusable labels |
| Question–program alignment | Question expresses the program semantics | Mismatched language and computation |
| Leakage prevention | Question does not reveal the answer | Self-play shortcuts |
| Certificate creation | Snapshot, signature, evidence, and verification are stored | Untraceable records |

### Execution-derived gold authority

**Certified program + designated graph snapshot + bounded executor → execution result → gold answer**

The model is not allowed to invent or copy a gold answer into the task proposal. The stored answer can therefore be regenerated from the program and graph state.

### Suggested task-certificate fields

| Field | Purpose |
|---|---|
| task_id | Stable task identifier |
| graph_snapshot | Identifies the graph state used for certification |
| question | Natural-language task |
| program | Executable graph semantics |
| program_signature | Canonical structural identity for deduplication |
| gold_answers | Result of certified execution |
| witness_facts | Graph facts supporting the result |
| operator_tags and program_cost | Structural description and complexity |
| verification_summary | Executability, alignment, leakage, and related checks |
| generation metadata | Round, seed context, and provenance |
| solver_statistics | Difficulty measurements |
| archive_decision | Accepted/rejected status and reason codes |

### Structured rejection categories

| Category | Example reason | Engineering value |
|---|---|---|
| Parse failure | NON_JSON, EXTRA_TEXT | Identifies output-contract failure |
| Schema failure | INVALID_SCHEMA, UNKNOWN_OP | Identifies unsupported structure |
| Relation failure | RELATION_NOT_ALLOWED | Identifies out-of-catalogue access |
| Budget failure | BUDGET_EXCEEDED | Identifies unsafe execution |
| Execution failure | EXECUTION_ERROR, EMPTY_RESULT | Identifies operationally invalid tasks |
| Semantic failure | Low question–program alignment | Identifies language mismatch |
| Leakage failure | Answer appears in the question | Prevents shortcut supervision |
| Archive failure | TOO_HARD, TOO_EASY, DUPLICATE_SIGNATURE, LOW_NOVELTY | Controls long-term distribution |

### Technical effect chain

**Joint question/program generation + bounded execution + certification checks + execution-derived gold → machine-verifiable task data with a reproducible answer source**

### Protection focus

The important combination is:

1. generate the question and executable graph program as a linked proposal;
2. validate and execute the program in a designated graph environment;
3. accept the task only after certification; and
4. derive the stored reference answer from that execution.

Certificates, witness facts, signatures, and rejection reasons provide useful dependent protection and practical detectability.

### Slide-ready text

- Questioner outputs **{question, executable graph program}** together.
- Certification checks format, grounding, budgets, execution, alignment, and leakage.
- Gold is generated only by executing the certified program.
- Accepted tasks receive a stable certificate; rejected tasks retain structured reasons.
- The resulting task data is reproducible, auditable, and deduplicable.

### Optional speaker notes

“The executor, not the task generator, acts as the source of truth. Otherwise self-evolution can amplify model mistakes. A certificate binds the question, program, graph snapshot, answer, evidence, and verification state into one replayable record.”
+

---

## Page 7 — Invention Point 3: Interface-Aware Curriculum Frontier Reward

### Core proposition

The Questioner is trained through production, grounding, and semantic-frontier stages. Solver interface readiness is measured separately from conditional semantic success and gates how strongly semantic difficulty contributes to the Questioner reward.

### Why a staged reward is required

Early in training, a Solver may fail because it cannot emit valid JSON, use the DSL schema, maintain handles, or execute a program. If all failures are treated as semantic failures, a malformed task can appear artificially difficult. The Questioner may then be rewarded for producing tasks the Solver cannot parse or execute.

The reward separates three questions:

1. Did the Questioner produce a complete machine-readable task?
2. Is the task grounded and executable in the graph environment?
3. Given that the Solver can use the interface, is the task near the desired semantic frontier?

### Stage definitions

| Stage | Representative milestones | Training purpose |
|---|---|---|
| Production | Question present, code present, JSON valid, schema fraction, valid operator fraction, valid prefix fraction | Learn the output contract first |
| Grounding | Seed coverage, valid relations/handles/types, executable prefix, execution, non-empty result, alignment, no leakage, certification | Learn to create real tasks in the graph environment |
| Frontier | Conditional semantic success after parsing and execution | Generate tasks near the Solver's capability boundary |

### Separated Solver signals

Let:

- **p_parse** be the fraction of Solver outputs that parse successfully;
- **p_exec|parse** be the fraction of parsed outputs that execute successfully;
- **p_sem|exec** be semantic success measured only among executable outputs.

The interface-readiness value is:

**I = p_parse × p_exec|parse**

A target-centred semantic-frontier reward may be:

**F = exp(−(p_sem|exec − τ)² / (2σ²))**

where τ is the target Solver success rate and σ controls the width of the preferred frontier.

The effective frontier weight is:

**w_frontier,eff = w_frontier,configured × I**

A representative frontier-stage total is:

**R = (1 − w_frontier,eff) × R_base + w_frontier,eff × F**

where R_base combines production and grounding progress. Equivalent monotonic gating functions may be used if they preserve the separation between interface readiness and semantic difficulty.

### Interpretation table

| Situation | Interface readiness | Conditional semantic success | Reward behavior |
|---|---:|---:|---|
| Solver cannot parse | Low | Not meaningful | Frontier contribution is strongly suppressed |
| Solver parses but cannot execute | Low to medium | Not meaningful | Grounding remains important |
| Solver executes but answers incorrectly | High | Low | The task is genuinely difficult |
| Solver executes and solves almost everything | High | High | The task may be too easy |
| Solver succeeds near the target | High | Near target | Frontier contribution is maximized |

### Dense reward advantages

- Provides learning signal before complete certification is frequent.
- Identifies progress in output production, graph grounding, or semantic task quality.
- Avoids a single sparse certified/not-certified reward.
- Avoids treating parser or executor failure as semantic difficulty.
- Produces structured reward components for logs and audit.

### Technical effect chain

**Milestone-based production/grounding scores + conditional Solver statistics + interface-readiness gate → better-aligned Questioner learning and a meaningful semantic frontier**

### Protection focus

Protect the cooperative relationship:

- interface competence is measured separately from semantic competence;
- semantic success is conditioned on successful execution;
- a frontier-related reward contribution is controlled by interface readiness; and
- the staged reward updates task generation.

The exact multiplication, Gaussian function, target value, sigma, or stage duration should normally be an embodiment or dependent limitation, not the only protected form.

### Slide-ready text

- Stage 1 learns the **question + program contract**.
- Stage 2 learns **graph grounding and executability**.
- Stage 3 learns **semantic frontier difficulty**.
- Interface readiness = parse rate × execution rate given parse.
- Semantic success is measured only on executable outputs.
- Interface readiness gates the frontier reward.

### Optional speaker notes

“A Solver that cannot use the program interface makes every task look hard. We therefore measure parsing, execution given parsing, and semantic success given execution. The semantic-frontier signal is activated only in proportion to interface readiness.”

---

## Page 8 — Invention Point 4: Dual-Role Same-Round Self-Evolution and Deterministic Task Management

### Core proposition

Questioner and Solver use separate role-specific adapters. Within a defined round, Questioner-generated tasks are certified and admitted before the Solver dataset is rebuilt, allowing the Solver to learn from newly admitted tasks in the same round. Archive admission and curriculum sampling control quality, novelty, difficulty, and replay.

### Role separation

| Mechanism | Technical purpose |
|---|---|
| Separate Questioner adapter | Preserves task-generation policy and reward history |
| Separate Solver adapter | Preserves solving capability and provides a stable reference |
| Frozen Solver during evaluation | Prevents the difficulty judge from moving during candidate scoring |
| Role-specific datasets | Prevents unintended mixing of Questioner and Solver supervision |
| Separate adapter IDs in round state | Makes the training path recoverable and auditable |

### Defined same-round order

**Questioner update → candidate generation/certification → deterministic archive admission → Solver dataset rebuild → Solver update**

This order permits newly accepted tasks from round r to enter Solver training in round r, rather than waiting until round r+1. The update order is stored in the round plan or manifest.

### Deterministic archive admission

Candidates are processed in a stable order and evaluated using:

| Dimension | Example implementation | Purpose |
|---|---|---|
| Difficulty | Frozen-Solver pass rate within a minimum/maximum window | Reject unusably hard or trivial tasks |
| Structural novelty | Canonical program signature differs from archived signatures | Prevent duplicate executable structures |
| Textual novelty | Token n-gram or other text/semantic similarity | Reduce paraphrase duplication |
| Certification | Valid certificate and execution-derived gold | Prevent invalid tasks entering the archive |
| Structured decision | accepted flag, values, thresholds, reason codes | Preserve an explainable record |

Representative rejection reasons include **TOO_HARD**, **TOO_EASY**, **DUPLICATE_SIGNATURE**, and **LOW_NOVELTY**.

### Curriculum dataset reconstruction

The Solver dataset may combine:

- base tasks that preserve foundational competence;
- historical archived tasks that retain previous skills;
- newly admitted tasks that expose the current frontier.

An easy-to-hard sampling boundary may expand across rounds. A controlled proportion of easier tasks may be replayed to reduce forgetting. Selection uses an explicit random seed so the round can be reconstructed.

### Replayable round state

| Stored artifact | Reason |
|---|---|
| Configuration hash | Detects incompatible changes during resume |
| Graph snapshot and relation catalogue ID | Reconstructs the execution environment |
| Explicit random seed | Reconstructs sampling and ordering |
| Questioner/Solver adapter paths or hashes | Identifies role-specific state |
| Dataset path, row counts, and hash | Identifies the exact training data |
| Candidate and persistent archives | Preserves the task lifecycle |
| Admission decisions and thresholds | Explains accepted and rejected tasks |
| Reward component logs | Explains optimization signals |
| Update order and completed-round index | Supports safe continuation |

### Technical effect chain

**Separate adapters + frozen evaluation role + deterministic archive + same-round rebuilding + replay curriculum → stable co-evolution with controlled task quality and recoverable state**

### Protection focus

The stronger distinction lies in the ordered interaction with certified tasks:

1. a Questioner generates and learns from certified task proposals;
2. a frozen Solver supplies separated difficulty signals;
3. accepted tasks are deterministically recorded;
4. the Solver dataset is rebuilt using the new archive state;
5. a separate Solver adapter is updated; and
6. both role states enter the next round.

### Slide-ready text

- Questioner and Solver maintain separate adapters and data.
- Candidate tasks are certified before archive admission.
- Admission uses difficulty, structural novelty, textual novelty, and reason codes.
- Newly admitted tasks can be consumed by the Solver in the same round.
- Base, historical, and new tasks are sampled through a replayable curriculum.
- Round manifests preserve graph, seed, dataset, adapter, reward, and archive state.

### Optional speaker notes

“The innovation is not simply two agents playing against each other. The value comes from the update order and the certified data contract between them. Tasks pass deterministic admission, enter the current Solver dataset, and are learned by a separate adapter. Every transition can be reconstructed.”

---

## Page 9 — Consolidated Four-Point Invention Chain

### Suggested slide headline

**Four Cooperating Invention Points Form One Protected Technical Chain**

**Bounded executable DSL → joint task generation and certification → interface-aware frontier reward → role-separated same-round self-evolution**

| Invention point | Distinguishing mechanism | Technical importance | Suggested protection level |
|---|---|---|---|
| 1. DSL-driven bounded interface | Typed operations, dependencies, relation restrictions, budgets, traces | Establishes a machine-action and verification boundary | Foundation of principal claim |
| 2. Joint generation and certification | Question/program proposal, execution-derived gold, certificate, rejection reasons | Converts model output into reliable task data | Central principal limitation |
| 3. Interface-aware frontier reward | Conditional semantic statistics and interface-gated frontier contribution | Prevents malformed tasks being rewarded as difficult | Strong dependent or combined core |
| 4. Same-round dual-role self-evolution | Separate adapters, frozen evaluation, deterministic admission, rebuilding, replay | Controls task distribution and role interference | Combined core and layered dependents |

### Features not to emphasize in isolation

The following are useful but less distinctive alone:

- execution traces;
- a task archive;
- easy-sample replay;
- two model roles;
- graph-query generation; and
- a target-based reward.

Their value arises from cooperation with certified executable tasks and the ordered self-evolution loop.

### Proposed principal protection chain

1. obtain a designated graph environment and executable-program contract;
2. generate a natural-language graph task and corresponding bounded program;
3. validate and execute the program;
4. derive and store a reference answer and certificate from execution;
5. evaluate the task using separated interface and semantic signals;
6. admit certified tasks according to deterministic policy;
7. rebuild role-specific training data and update separate role states; and
8. output an updated graph agent and controlled certified-task archive.

### Fallback protection layers

| Layer | Example fallback limitation |
|---|---|
| Program contract | SSA handles, relation whitelist, terminal emit, operator profile |
| Bounded execution | Edge-visit and returned-entity budgets |
| Certification | Alignment, non-empty answer, no leakage, witness facts |
| Reward | Parse rate × conditional execution rate; conditional semantic success |
| Admission | Pass-rate window plus structural and textual novelty |
| Update order | Questioner → archive → Solver within the same round |
| State artifact | Separate adapter IDs, dataset hashes, round manifest |
| Service output | Program object, trace, certificate, and structured error |

### Recommended visual

Use **figures/fig3_patent_method_flow.png**.
+

---

# Part III. Industrial Applicability

## Page 10 — Industrial Applications

### General applicability

The method is applicable wherever a graph agent interacts with structured relational data and where an answer, generated task, or training record should be executable, bounded, and auditable. It may be deployed as a training platform, inference service, task-data service, or governance layer.

### Application matrix

| Scenario | Example graph data | Example executable task | Concrete output | Industrial value |
|---|---|---|---|---|
| Enterprise knowledge management | Products, customers, contracts, teams, policies | Resolve an entity, follow responsibility relations, filter by date | Entity/relationship result with evidence | Private-knowledge QA and audit |
| Financial risk and compliance | Ownership, transactions, accounts, counterparties, rules | Traverse ownership paths and verify a threshold condition | Risk entity, path, and rule-check result | Controlled multi-hop penetration |
| Biomedical research | Diseases, drugs, targets, trials, publications | Find candidates through defined relations and filter by condition | Candidate relationship with witness facts | Traceable research retrieval; not clinical advice |
| Industrial maintenance | Equipment, components, sensors, faults, repairs | Follow fault dependencies and compare attributes | Affected component or maintenance candidate | Dynamic private operational knowledge |
| Cybersecurity | Assets, vulnerabilities, alerts, identities, attack paths | Traverse allowed attack relations under an edge budget | Affected assets and bounded path evidence | Explainable investigation |
| Data governance | Tables, fields, metrics, lineage, permissions | Follow lineage and identify downstream impact | Lineage path, impact scope, or permission relation | Change and compliance audit |
| Recommendation | Users, items, interactions, categories, constraints | Intersect preference paths and apply eligibility filters | Candidate items with relation evidence | Controllable graph recommendation |
| Supply-chain analysis | Suppliers, parts, facilities, shipments, incidents | Traverse dependencies and filter by region or time | Exposed suppliers, parts, or facilities | Multi-tier risk visibility |
| Education and assessment | Concepts, prerequisites, outcomes, exercises | Generate a question whose answer is produced by a graph program | Difficulty-controlled certified exercise | Automated task-bank construction |
| Software/data engineering | Services, APIs, repositories, dependencies, incidents | Trace dependencies and locate impacted components | Impacted service list and trace | Incident response and migration planning |

### Deployment forms

| Deployment form | Description | Main deliverables |
|---|---|---|
| Self-evolution training platform | Generates, certifies, selects, and trains on organization-specific tasks | Updated adapters, certified archive, manifests |
| Graph-agent inference service | Converts a request into a bounded graph program and executes it | Answer, program, evidence, trace, budget usage |
| Certified task-data service | Produces graph tasks for SFT, RL, evaluation, or benchmarking | Question, program, gold, certificate, provenance |
| Agent governance layer | Validates program structure and graph-access limits | Admission/rejection, error code, audit trace |
| On-premises graph intelligence system | Runs against private graph data under local policies | Local outputs and auditable records |

### Example implementation narrative

In a financial compliance deployment, an organization provides a versioned graph snapshot containing companies, shareholders, accounts, transactions, and regulatory rules. The Questioner produces a natural-language investigation task and typed graph program. The executor restricts traversal to an approved relation catalogue and enforces an edge-visit budget. The resulting risk subject and relationship path are stored with a task certificate. Executable, non-duplicative tasks within a selected Solver difficulty window enter the archive. The Solver learns from those tasks without requiring an external annotation service to access the private graph.

### Practical benefits

- Reduces dependence on manually authored graph queries and labels.
- Supports organization-specific schemas and private graph snapshots.
- Produces evidence-bearing outputs suitable for inspection.
- Controls graph access and resource consumption.
- Diagnoses failures by stage rather than by one final score.
- Supports incremental task-library growth without accepting every generated sample.
- Enables reproducible offline training and controlled online inference.

### Slide-ready text

- Enterprise knowledge and private-data QA
- Financial relationship, risk, and compliance analysis
- Biomedical and scientific knowledge exploration
- Industrial maintenance and supply-chain dependency analysis
- Cybersecurity path investigation
- Data lineage, governance, and impact analysis
- Certified task generation for training and assessment

### Optional speaker notes

“The method is not limited to one benchmark. Its applicability follows from the contract: a graph backend, relation catalogue, executable DSL, and certification loop. Any industry with structured relationship data and a need for traceable outputs can adopt the architecture.”

---

# Part IV. Detectability and Protection Strategy

## Page 11 — Detectability

### Definition

Detectability means whether a technical feature could be observed or reasonably inferred from a competing product's documented interfaces, outputs, error behavior, exported artifacts, logs, or lawful black-box testing. It is not a legal conclusion regarding infringement or claim construction.

### Detectability statement

The invention should be drafted so that a meaningful portion of the protected technical chain is externally observable. The inference-side program interface, validation behavior, bounded execution, evidence output, and task-certificate artifacts are relatively detectable. Exact reward formulas, adapter separation, archive thresholds, and same-round schedules are more likely to remain internal.

Accordingly, the principal protection should include observable input–operation–output features, while internal self-evolution mechanisms are preserved as layered dependent or alternative limitations. The strongest differentiation arises from their combination, but enforcement should not depend only on a hidden reward formula.

### Detectability matrix

| Feature | Detectability | Possible evidence source | Protection implication |
|---|---|---|---|
| Natural-language input is converted into a graph program | High | API, SDK object, debug view, exported plan, documentation | Include in the core inference chain |
| Program contains graph operators, relations, directions, handles, and output | High | Program payload, query log, client object | Define the program contract |
| Program is validated before execution | Medium–High | Validation errors, rejected payloads, status fields | Claim validation and structured rejection |
| Graph access is bounded | Medium–High | Limit errors, trace metadata, configuration, repeated tests | Include budget enforcement |
| Result includes evidence or execution trace | High | UI evidence panel, API trace, audit report | Claim concrete trace/evidence output |
| Gold is derived only by certified execution | Medium | Dataset schema, certificate export, training artifacts | Protect as a training limitation |
| Certificate binds question, program, snapshot, answer, evidence | Medium–High | Exported task record or audit endpoint | Useful observable dependent limitation |
| Solver signals separate parsing, execution, and semantics | Low–Medium | Metrics, source code, training logs | Preserve as internal-method protection |
| Interface readiness gates frontier reward | Low | Reward logs, code, experiment records | Do not make this the sole enforcement anchor |
| Tasks are admitted by difficulty and novelty | Medium | Archive metadata, rejection reasons, task behavior | Claim decision records and reason codes |
| Questioner and Solver use separate adapters | Low–Medium | Manifests, checkpoints, documentation | Important dependent claim |
| New tasks are consumed in the same round | Low–Medium | Timestamps, dataset hashes, manifests | Tie to stored round-state artifacts |
| State is replayable from snapshot, seed, data, and adapter IDs | Medium | Manifest and reproducibility package | Claim the stored state record |

### Observable artifacts that strengthen protection

- Versioned executable graph-program objects.
- Explicit relation identifiers, directions, handles, and terminal output fields.
- Structured parse, schema, relation, execution, and budget error codes.
- Per-step traces with operators, handle state, evidence, and cumulative usage.
- Certificates with program signature, graph snapshot, execution-derived answer, and witness facts.
- Admission records with difficulty, novelty, accepted status, and reason codes.
- Round manifests with separate role-adapter identifiers and dataset hashes.
- Outputs that distinguish model-generated content from executor-derived results.

### Potential lawful black-box observations

| Test | Possible observation |
|---|---|
| Submit a valid and invalid operator | Distinct schema or operator rejection |
| Use an out-of-catalogue relation | Relation-whitelist rejection |
| Increase traversal depth or result size | Stable budget error or bounded behavior |
| Repeat semantically identical tasks | Duplicate or novelty-related archive behavior |
| Request debug or audit output | Program, trace, evidence, or certificate |
| Change the graph snapshot | Evidence that answers depend on external execution |
| Inspect batch task exports | Linked question/program/gold/certificate fields |

Such tests should only be performed against systems and data for which the tester has lawful authorization.

### Recommended layered claim strategy

1. **Observable principal inference chain**
   Natural-language task → bounded typed graph program → validation → graph execution → concrete graph-domain result.

2. **Certified task-generation layer**
   Joint question/program generation → execution-derived reference answer → stored certificate and evidence.

3. **Interface-aware learning layer**
   Separate parsing, conditional execution, and conditional semantic signals → interface-gated frontier contribution.

4. **Self-evolution and task-management layer**
   Separate role states → deterministic admission → same-round dataset rebuilding → updated Solver and Questioner states.

5. **Stored-artifact layer**
   Program object, execution trace, certificate, admission record, and round manifest.

6. **System/service embodiments**
   Training platform, inference service, certified task-data service, and graph-agent governance layer, where supported.

### Drafting risks to avoid

- Do not claim only “using a DSL”; DSLs and program synthesis are broad known concepts.
- Do not claim only “Questioner/Solver self-play”; role-based self-play is broad.
- Do not depend only on an exact reward formula that may be hidden or altered.
- Do not make unsupported accuracy, cost, latency, or state-of-the-art claims.
- Do not unnecessarily limit the principal claim to GraphScript v0.3, KQAPro, Qwen, GRPO, LoRA, SQLite, or exact thresholds.
- Do not describe a software output only as a “result”; identify a graph answer, evidence path, certified task, updated adapter, or controlled archive.

### Short conclusion

**The most detectable protection anchor is the bounded executable graph-program interface and its traceable outputs. The strongest differentiation is the certified self-evolution chain built on that interface.**

### Optional speaker notes

“A competitor's internal training code may be difficult to inspect, so the patent should not rely only on the curriculum formula. The executable program, validation behavior, bounded graph execution, evidence, and certificates are more observable. Internal mechanisms can then be protected as dependent layers connected to those artifacts.”

---

# Optional Final Summary Slide

## Suggested headline

**A Verifiable Path from Graph Access to Graph-Agent Self-Evolution**

| Prior limitation | Proposed mechanism | Technical effect |
|---|---|---|
| Free-form or fragmented graph interaction | Typed bounded executable DSL | Stable action and verification interface |
| Unverified generated tasks and labels | Joint generation with execution-derived gold | Certified and reproducible task data |
| Interface failure confused with difficulty | Interface-aware staged frontier reward | Better-aligned self-play signal |
| Mixed roles and uncontrolled task accumulation | Separate adapters, deterministic archive, same-round rebuilding | Controlled and replayable self-evolution |

### Final one-sentence message

**The invention uses executable graph programs as the common technical contract through which a Graph Agent can generate, verify, manage, and learn from its own graph tasks.**

---

# Use and Review Notes

- This material is based on the current project implementation and configuration.
- It intentionally contains more text than a final PPT so the inventor can select and shorten it.
- No unverified quantitative improvement or state-of-the-art claim is included.
- A dedicated search should examine executable-code KGQA, graph-path supervision, Questioner/Solver self-play, persistent-memory skill evolution, and knowledge-graph co-evolution.
- Final claim scope, inventorship, ownership, publication dates, and filing strategy require inventor confirmation and review by a qualified patent professional.
- This is presentation and technical-disclosure material, not a filing-ready patent application or legal opinion.
