# Figure QA notes

## Automated source preflight

- Backend: Python only for Fig. 1 and Fig. 3.
- Result: 13 PASS, 1 WARN, 0 FAIL.
- The warning concerns the 338.7 mm width. This is intentional because the target is a 16:9 PPT slide, not an 89/183 mm journal column.
- SVG editable text, PDF Type 42 fonts, 300 dpi PNG and 600 dpi TIFF are configured.

## Visual inspection

- Fig. 1: all three paradigm columns, arrows and notes fit within their containers after wrapping.
- Fig. 2 English v2: generated with built-in GPT Image and inspected for module completeness, English-only labels, legibility, and role-update direction. The recommended file is `fig2_curriculum_v3_architecture_gpt_image_en_v2.png`, intended for a white 16:9 slide. The v2 correction removes the candidate-task-to-Questioner update implication and preserves reward feedback as the Questioner update path. AI-rendered labels still require final human verification before external filing or publication.
- Fig. 3: all seven ordered steps and concrete outputs are visible; arrows follow the claimed data flow. This remains a draft until formal claim terminology is fixed.

## Integrity and evidence

- No mock performance values or quantitative superiority claims are shown.
- Figures represent source-supported software operations. The comparison figure distinguishes mechanism-level mitigation from experimentally demonstrated gains.
