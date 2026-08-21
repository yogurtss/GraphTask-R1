# GraphTask-R1 Patent Presentation Material Pack

This directory contains English-language source material for preparing a patent-introduction PPT. It is deliberately more detailed than a final slide deck so that the inventor can select and shorten the content. It is not a claim set, a filing-ready patent application, or a legal opinion.

## Recommended reading order

1. **01_patent_presentation_materials_en.md** — complete four-part presentation source:
   - Technical Field of the Invention;
   - Description of the Prior Art;
   - Problems of the Prior Art;
   - Summary of the Invention;
   - four consolidated invention points;
   - Industrial Applicability;
   - Detectability and protection strategy.
2. **figures/fig1_related_work_comparison.png** — prompt-based, tool-based, and executable-interface comparison.
3. **figures/fig2_dsl_adversarial_coevolution_gpt_image_v2.png** — recommended GPT Image architecture diagram with a balanced mix of concise technical labels and restrained visual detail.
4. **figures/fig3_patent_method_flow.png** — black-and-white claim-aligned method-flow draft.
5. **02_technical_evidence_and_confirmations_en.md** — evidence ledger and inventor-confirmation questions.
6. **03_related_work_and_search_notes_en.md** — related-work and search starting points.

## Figure notes

- Figures 1 and 3 are generated deterministically with Python/Matplotlib and are available as SVG/PDF plus PNG/TIFF.
- The recommended Figure 2 was generated with GPT Image for a white 16:9 slide and should be manually checked before external use. The editable SVG/PDF version remains available as a structural fallback.
- Earlier lane-heavy AI architecture images are retained only as references.
- Figure labels and method order must be aligned with final claim terminology before filing.

## Recommended title

**DSL-Driven Self-Evolving Graph Agent**

Formal alternative: **A Method for DSL-Driven Self-Evolution of a Graph Agent**

## One-sentence concept

A Questioner jointly generates a natural-language task and a bounded typed graph program; certified execution produces the reference answer; interface-aware curriculum rewards, deterministic archive admission, and separate Questioner/Solver updates form a replayable same-round self-evolution loop.
