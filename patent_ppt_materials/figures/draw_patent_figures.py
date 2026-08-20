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
    draw_patent_flow()
