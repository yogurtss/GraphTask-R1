from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parent

NAVY = "#17324D"
BLUE = "#4C78A8"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
SLATE = "#5C677D"
LIGHT_BLUE = "#EAF2F8"
LIGHT_TEAL = "#E8F5F2"
LIGHT_AMBER = "#FFF4DE"
LIGHT_GRAY = "#F3F5F7"
WHITE = "#FFFFFF"
BLACK = "#111111"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 12,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = WHITE,
    edgecolor: str = NAVY,
    textcolor: str = NAVY,
    fontsize: float = 12,
    linewidth: float = 1.5,
    radius: float = 0.018,
    weight: str = "normal",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=textcolor,
        fontsize=fontsize,
        weight=weight,
        linespacing=1.3,
    )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY,
    linewidth: float = 1.7,
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=14,
            color=color,
            linewidth=linewidth,
            connectionstyle=connectionstyle,
        )
    )


def save_bundle(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor=WHITE)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor=WHITE)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(
        OUTPUT_DIR / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor=WHITE,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def draw_related_work() -> None:
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(
        0.5,
        0.955,
        "Graph reasoning paradigms: where failures occur",
        ha="center",
        va="top",
        fontsize=22,
        weight="bold",
        color=NAVY,
    )
    ax.text(
        0.5,
        0.905,
        "Prompt-based and tool-based methods access graph knowledge, but expose different control and verification gaps",
        ha="center",
        va="top",
        fontsize=10.5,
        color=SLATE,
    )

    columns = [
        (
            0.03,
            "A  Direct prompting",
            LIGHT_BLUE,
            BLUE,
            ["Question + flattened KG context", "LLM free-form reasoning", "Free-text answer"],
            [
                "Structure diluted in text",
                "Context truncation / prompt sensitivity",
                "Hallucinated joins or parametric recall",
                "Answer is not forced through execution",
            ],
        ),
        (
            0.345,
            "B  Iterative tool agent",
            LIGHT_AMBER,
            AMBER,
            ["Plan", "Call tool", "Observe", "Repeat / answer"],
            [
                "Latency and cost grow with steps",
                "Invalid calls and early-error propagation",
                "Variable-length, framework-coupled traces",
                "Sparse or ambiguous credit assignment",
            ],
        ),
        (
            0.66,
            "C  GraphTask executable interface",
            LIGHT_TEAL,
            TEAL,
            ["Question + catalogue", "One-shot typed GraphScript", "Bounded executor", "Answer + trace"],
            [
                "Schema and relation whitelist",
                "Operator / edge / return budgets",
                "Execution-derived answer",
                "Structured rejection and replayable trace",
            ],
        ),
    ]

    for left, title, fill_color, accent, flow, notes in columns:
        width = 0.29
        container = FancyBboxPatch(
            (left, 0.11),
            width,
            0.74,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=fill_color,
            edgecolor=accent,
            linewidth=1.7,
        )
        ax.add_patch(container)
        ax.text(
            left + 0.018,
            0.815,
            fill(title, width=32),
            fontsize=12.5,
            weight="bold",
            color=NAVY,
            va="center",
            linespacing=1.05,
        )

        n = len(flow)
        box_h = 0.062
        start_y = 0.69
        gap = 0.024
        ys: list[float] = []
        for index, label in enumerate(flow):
            y = start_y - index * (box_h + gap)
            ys.append(y)
            add_box(
                ax,
                (left + 0.035, y),
                width - 0.07,
                box_h,
                label,
                facecolor=WHITE,
                edgecolor=accent,
                textcolor=NAVY,
                fontsize=9.3,
                linewidth=1.3,
                radius=0.012,
                weight="bold" if index == 1 else "normal",
            )
            if index > 0:
                previous_y = ys[index - 1]
                add_arrow(
                    ax,
                    (left + width / 2, previous_y - 0.006),
                    (left + width / 2, y + box_h + 0.006),
                    color=accent,
                    linewidth=1.5,
                )

        notes_y = 0.32
        ax.plot(
            [left + 0.03, left + width - 0.03],
            [notes_y + 0.035, notes_y + 0.035],
            color=accent,
            linewidth=1.0,
            alpha=0.65,
        )
        for index, note in enumerate(notes):
            bullet_color = accent if left >= 0.66 else "#B24C3D"
            y = notes_y - index * 0.059
            ax.text(left + 0.042, y, "•", fontsize=12, color=bullet_color, va="center")
            ax.text(
                left + 0.062,
                y,
                fill(note, width=36),
                fontsize=8.0,
                color=NAVY,
                va="center",
                linespacing=1.12,
            )

    ax.text(
        0.5,
        0.045,
        "Design target: a verifiable, bounded and replayable interface between language generation and graph execution",
        ha="center",
        fontsize=10.5,
        weight="bold",
        color=NAVY,
    )
    save_bundle(fig, "fig1_related_work_comparison")
    plt.close(fig)


def draw_curriculum_architecture() -> None:
    """Draw a DSL-first, left-to-right view of Curriculum v3 co-evolution."""
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(
        0.5,
        0.965,
        "GraphTask-R1 Curriculum v3: DSL-Centered Adversarial Co-Evolution",
        ha="center",
        va="top",
        fontsize=21,
        weight="bold",
        color=NAVY,
    )
    ax.text(
        0.5,
        0.918,
        "A Questioner proposes executable challenges; certified execution supplies gold; a Solver attempts and learns from them",
        ha="center",
        va="top",
        fontsize=9.7,
        color=SLATE,
    )

    # The curriculum remains visible, but is deliberately subordinate to the
    # sample-level DSL and learning loop below.
    ax.text(
        0.105,
        0.868,
        "CURRICULUM\nACROSS ROUNDS",
        ha="center",
        fontsize=7.1,
        weight="bold",
        color=SLATE,
        va="center",
        linespacing=1.05,
    )
    stage_specs = [
        (0.205, "1  PRODUCTION", "question + program"),
        (0.465, "2  GROUNDING", "typed, executable, aligned"),
        (0.725, "3  FRONTIER", "conditional semantic difficulty"),
    ]
    for left, heading, detail in stage_specs:
        add_box(
            ax,
            (left, 0.838),
            0.24,
            0.058,
            f"{heading}\n{detail}",
            facecolor=LIGHT_GRAY,
            edgecolor="#B8C4D1",
            textcolor=NAVY,
            fontsize=7.5,
            linewidth=1.0,
            radius=0.01,
            weight="bold",
        )

    # Column 1: instance-scoped context and the current Questioner.
    add_box(
        ax,
        (0.025, 0.33),
        0.155,
        0.45,
        "",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
        linewidth=1.5,
        radius=0.014,
    )
    ax.text(
        0.1025,
        0.745,
        "QUESTIONER  $Q_t$",
        ha="center",
        va="center",
        fontsize=10.2,
        weight="bold",
        color=NAVY,
    )
    ax.text(
        0.1025,
        0.708,
        "proposes one challenge",
        ha="center",
        va="center",
        fontsize=8.0,
        color=BLUE,
        weight="bold",
    )
    ax.plot([0.045, 0.16], [0.681, 0.681], color="#B8CDE0", linewidth=1.0)
    ax.text(
        0.047,
        0.648,
        "ANSWER-FREE CONTEXT",
        ha="left",
        va="center",
        fontsize=7.8,
        weight="bold",
        color=SLATE,
    )
    for index, item in enumerate(
        [
            "graph snapshot",
            "relation catalogue",
            "required seed IDs",
            "explicit random seed",
            "bounded budgets",
        ]
    ):
        y = 0.607 - index * 0.047
        ax.text(0.05, y, "•", fontsize=10, color=BLUE, va="center")
        ax.text(0.064, y, item, fontsize=7.7, color=NAVY, va="center")
    ax.text(
        0.1025,
        0.361,
        "instance-scoped • replayable",
        ha="center",
        va="center",
        fontsize=7.0,
        color=SLATE,
    )

    # Column 2: hero panel — the actual current DSL envelope.
    add_box(
        ax,
        (0.215, 0.285),
        0.31,
        0.535,
        "",
        facecolor=WHITE,
        edgecolor=TEAL,
        linewidth=2.1,
        radius=0.016,
    )
    ax.text(
        0.235,
        0.787,
        "CURRENT DSL TASK CONTRACT",
        ha="left",
        va="center",
        fontsize=10.5,
        weight="bold",
        color=TEAL,
    )
    code_text = (
        '{\n'
        '  "question": "... ?",\n'
        '  "program": {\n'
        '    "ops": [\n'
        '      {"op":"resolve_entity",\n'
        '       "query":"SEED_ID", ...},\n'
        '      {"op":"follow",\n'
        '       "relation":"CATALOG_ID", ...},\n'
        '      ...\n'
        '      {"op":"emit", ...}\n'
        '    ]\n'
        '  }\n'
        '}'
    )
    ax.text(
        0.238,
        0.744,
        code_text,
        ha="left",
        va="top",
        family="monospace",
        fontsize=6.6,
        color=NAVY,
        linespacing=1.18,
    )
    dsl_rules = ["question + program", "catalog-constrained ops", "executable"]
    chip_widths = [0.08, 0.105, 0.07]
    chip_x = 0.232
    for label, width in zip(dsl_rules, chip_widths, strict=True):
        add_box(
            ax,
            (chip_x, 0.32),
            width,
            0.038,
            label,
            facecolor=LIGHT_TEAL,
            edgecolor="#8CC9BF",
            textcolor=NAVY,
            fontsize=5.8,
            linewidth=0.9,
            radius=0.007,
            weight="bold",
        )
        chip_x += width + 0.008
    ax.text(
        0.37,
        0.295,
        "No answer field — gold is never proposed by the model",
        ha="center",
        va="center",
        fontsize=6.9,
        weight="bold",
        color="#A25522",
    )

    # Column 3: one certified-execution gate replaces several competing lanes.
    add_box(
        ax,
        (0.56, 0.35),
        0.17,
        0.41,
        "",
        facecolor=LIGHT_AMBER,
        edgecolor=AMBER,
        linewidth=1.7,
        radius=0.014,
    )
    ax.text(
        0.645,
        0.724,
        "CERTIFIED EXECUTION",
        ha="center",
        va="center",
        fontsize=9.4,
        weight="bold",
        color=NAVY,
    )
    cert_steps = [
        ("1", "Parse + type/schema", "valid operators and fields"),
        ("2", "Bounded executor", "budgets + step trace"),
        ("3", "Execution-derived gold", "non-empty + aligned"),
    ]
    for index, (number, heading, detail) in enumerate(cert_steps):
        y = 0.644 - index * 0.092
        ax.text(
            0.584,
            y,
            number,
            ha="center",
            va="center",
            fontsize=7.2,
            weight="bold",
            color=WHITE,
            bbox={
                "boxstyle": "circle,pad=0.28",
                "facecolor": AMBER,
                "edgecolor": AMBER,
                "linewidth": 0.8,
            },
        )
        ax.text(0.605, y + 0.012, heading, fontsize=7.5, weight="bold", color=NAVY, va="center")
        ax.text(0.605, y - 0.018, detail, fontsize=6.5, color=SLATE, va="center")
    ax.plot([0.58, 0.71], [0.407, 0.407], color="#E5C17F", linewidth=1.0)
    ax.text(
        0.645,
        0.381,
        "certificate + trace",
        ha="center",
        fontsize=7.0,
        weight="bold",
        color=NAVY,
    )
    ax.text(
        0.645,
        0.356,
        "structured rejection or deterministic admission",
        ha="center",
        fontsize=5.9,
        color=SLATE,
    )

    # Column 4: both adversarial pressure and cooperative updates live together.
    add_box(
        ax,
        (0.765, 0.285),
        0.21,
        0.535,
        "",
        facecolor=LIGHT_GRAY,
        edgecolor=NAVY,
        linewidth=1.8,
        radius=0.016,
    )
    ax.text(
        0.87,
        0.787,
        "ADVERSARIAL CO-EVOLUTION",
        ha="center",
        va="center",
        fontsize=9.8,
        weight="bold",
        color=NAVY,
    )
    add_box(
        ax,
        (0.787, 0.674),
        0.166,
        0.074,
        "Frozen Solver  $S_t$\nattempts the question",
        facecolor=WHITE,
        edgecolor=BLUE,
        textcolor=NAVY,
        fontsize=7.5,
        linewidth=1.2,
        radius=0.01,
        weight="bold",
    )
    ax.text(
        0.87,
        0.645,
        "cannot see the Questioner's program or gold",
        ha="center",
        fontsize=6.2,
        color=SLATE,
    )
    add_box(
        ax,
        (0.787, 0.535),
        0.166,
        0.078,
        "CONDITIONAL DUEL\nsemantic success | Solver executes",
        facecolor=LIGHT_AMBER,
        edgecolor=AMBER,
        textcolor=NAVY,
        fontsize=7.0,
        linewidth=1.1,
        radius=0.01,
        weight="bold",
    )
    add_arrow(ax, (0.87, 0.668), (0.87, 0.62), color=SLATE, linewidth=1.2)
    ax.text(
        0.87,
        0.504,
        "role-separated updates",
        ha="center",
        fontsize=6.8,
        color=SLATE,
        weight="bold",
    )
    add_box(
        ax,
        (0.783, 0.405),
        0.08,
        0.084,
        "Questioner\n$Q_{t+1}$\nstaged reward",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
        textcolor=NAVY,
        fontsize=6.5,
        linewidth=1.1,
        radius=0.009,
        weight="bold",
    )
    add_box(
        ax,
        (0.877, 0.405),
        0.08,
        0.084,
        "Solver\n$S_{t+1}$\nadmitted tasks",
        facecolor=LIGHT_TEAL,
        edgecolor=TEAL,
        textcolor=NAVY,
        fontsize=6.5,
        linewidth=1.1,
        radius=0.009,
        weight="bold",
    )
    add_arrow(ax, (0.85, 0.53), (0.823, 0.496), color=BLUE, linewidth=1.1)
    add_arrow(ax, (0.89, 0.53), (0.917, 0.496), color=TEAL, linewidth=1.1)
    ax.text(
        0.87,
        0.363,
        "same round: admit → rebuild Solver data → update",
        ha="center",
        fontsize=6.0,
        color=SLATE,
    )
    ax.text(
        0.87,
        0.326,
        "controlled task archive\n+ replayable round state",
        ha="center",
        fontsize=6.1,
        weight="bold",
        color=NAVY,
        linespacing=1.05,
    )

    # One clean left-to-right spine.
    add_arrow(ax, (0.183, 0.555), (0.211, 0.555), color=NAVY, linewidth=1.8)
    add_arrow(ax, (0.528, 0.555), (0.556, 0.555), color=NAVY, linewidth=1.8)
    add_arrow(ax, (0.733, 0.555), (0.761, 0.555), color=NAVY, linewidth=1.8)
    ax.text(0.197, 0.575, "emits", ha="center", fontsize=6.2, color=SLATE)
    ax.text(0.542, 0.575, "execute", ha="center", fontsize=6.2, color=SLATE)
    ax.text(0.747, 0.575, "challenge", ha="center", fontsize=6.2, color=SLATE)

    # A single non-crossing feedback path explains the next-round co-evolution.
    ax.plot(
        [0.92, 0.92, 0.1025, 0.1025],
        [0.278, 0.225, 0.225, 0.315],
        color=NAVY,
        linewidth=1.55,
        solid_capstyle="round",
    )
    add_arrow(ax, (0.1025, 0.315), (0.1025, 0.326), color=NAVY, linewidth=1.55)
    add_box(
        ax,
        (0.285, 0.19),
        0.47,
        0.055,
        "NEXT ROUND: a stronger Solver raises the frontier • a stronger Questioner proposes harder certified tasks",
        facecolor=WHITE,
        edgecolor=NAVY,
        textcolor=NAVY,
        fontsize=7.3,
        linewidth=1.1,
        radius=0.01,
        weight="bold",
    )
    ax.text(
        0.5,
        0.115,
        "Outputs: verified graph-QA model  •  execution-derived task certificates and traces  •  controlled training-task archive",
        ha="center",
        va="center",
        fontsize=8.6,
        color=NAVY,
        weight="bold",
    )

    save_bundle(fig, "fig2_curriculum_v3_architecture_dsl_coevolution")
    plt.close(fig)


def draw_patent_flow() -> None:
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(
        0.5,
        0.96,
        "Claim-aligned method flow (draft)",
        ha="center",
        va="top",
        fontsize=23,
        weight="bold",
        color=BLACK,
    )

    steps = [
        ("S1", "Obtain graph snapshot, relation catalogue,\nseed set, base tasks, explicit random seed,\nand execution budgets"),
        ("S2", "Jointly generate a natural-language question\nand a bounded typed graph program"),
        ("S3", "Validate and execute the graph program;\nderive gold only from certified execution"),
        ("S4", "Evaluate with a frozen Solver; compute staged\nreward and gate the semantic frontier\nby interface readiness"),
        ("S5", "Admit certified tasks deterministically\nby difficulty window and structural /\ntextual novelty"),
        ("S6", "Rebuild the Solver dataset in the same round;\napply easy-to-hard sampling and easy replay"),
        ("S7", "Update separate Questioner and Solver adapters;\nstore replayable round state and traces"),
    ]
    x_positions = [0.04, 0.36, 0.68, 0.68, 0.36, 0.04, 0.36]
    y_positions = [0.72, 0.72, 0.72, 0.43, 0.43, 0.43, 0.16]
    box_w = 0.27
    box_h = 0.15

    for (step_id, label), x, y in zip(steps, x_positions, y_positions, strict=True):
        add_box(
            ax,
            (x, y),
            box_w,
            box_h,
            f"{step_id}\n{label}",
            facecolor=WHITE,
            edgecolor=BLACK,
            textcolor=BLACK,
            fontsize=8.2,
            linewidth=1.45,
            radius=0.008,
            weight="normal",
        )

    connectors = [
        ((0.31, 0.795), (0.36, 0.795)),
        ((0.63, 0.795), (0.68, 0.795)),
        ((0.815, 0.72), (0.815, 0.58)),
        ((0.68, 0.505), (0.63, 0.505)),
        ((0.36, 0.505), (0.31, 0.505)),
        ((0.63, 0.235), (0.68, 0.235)),
    ]
    for start, end in connectors:
        add_arrow(ax, start, end, color=BLACK, linewidth=1.5)
    ax.plot(
        [0.175, 0.175, 0.495],
        [0.43, 0.34, 0.34],
        color=BLACK,
        linewidth=1.5,
        solid_capstyle="round",
    )
    add_arrow(ax, (0.495, 0.34), (0.495, 0.31), color=BLACK, linewidth=1.5)

    add_box(
        ax,
        (0.68, 0.15),
        0.27,
        0.17,
        "Concrete outputs\nVerified graph-QA model\nCertified task archive + execution traces",
        facecolor=LIGHT_GRAY,
        edgecolor=BLACK,
        textcolor=BLACK,
        fontsize=8.8,
        linewidth=1.5,
        radius=0.008,
        weight="bold",
    )
    ax.text(
        0.5,
        0.045,
        "Every material step must remain consistent with the final claim terminology and implementation evidence.",
        ha="center",
        fontsize=10.5,
        color=BLACK,
    )
    save_bundle(fig, "fig3_patent_method_flow")
    plt.close(fig)


if __name__ == "__main__":
    draw_related_work()
    draw_curriculum_architecture()
    draw_patent_flow()
