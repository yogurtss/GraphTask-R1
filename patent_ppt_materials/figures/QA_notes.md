# Figure QA notes

## Automated source preflight

- Backend: Python only for Fig. 1 and Fig. 3.
- Result: 13 PASS, 1 WARN, 0 FAIL.
- The warning concerns the 338.7 mm width. This is intentional because the target is a 16:9 PPT slide, not an 89/183 mm journal column.
- SVG editable text, PDF Type 42 fonts, 300 dpi PNG and 600 dpi TIFF are configured.

## Visual inspection

- Fig. 1: all three paradigm columns, arrows and notes fit within their containers after wrapping.
- Fig. 2: generated with built-in GPT Image; inspected for module completeness and arrow direction. The PNG is 1672 × 941 RGBA and is intended for a white slide background. AI-rendered Chinese labels still require a final human spelling check before external filing or publication.
- Fig. 3: all seven ordered steps and concrete outputs are visible; arrows follow the claimed data flow. This remains a draft until formal claim terminology is fixed.

## Integrity and evidence

- No mock performance values or quantitative superiority claims are shown.
- Figures represent source-supported software operations. The comparison figure distinguishes mechanism-level mitigation from experimentally demonstrated gains.
