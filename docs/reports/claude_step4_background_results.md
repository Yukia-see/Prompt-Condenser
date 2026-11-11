# Step 4 results: background scan with characters masked out (Claude → Gemini, 2026-11-11)

Built as an option: the main figures (detector boxes ≥ 30% of the biggest) painted over in neutral grey, the rest
tagged once, only scene tags kept (concrete objects ≥ 0.40, setting / light ≥ 0.45). Measured with the tagger +
cleanup rules, threshold 0.30, 11 images:

| tagging | F1 | precision | time / image |
|---|---|---|---|
| one pass | 79.2% | 88% | 4.7 s |
| one pass + background pass | 79.2% | 88% | 7.4 s |
| full image + body zones | 79.2% | 86% | 15.3 s |
| full image + body zones + background | 79.2% | 86% | 17.8 s |
| today's full set (with the 4 corner crops) | 78.8% | 83% | 26.1 s |

The background pass added one tag across all 11 images (`brick wall` on rouge_bat, wrong): on these pictures the
characters fill most of the frame and the full-image pass already sees what background is left. **Not adopted; the
code was removed** (commit `f2068a0` records it).

Side result: the 4 corner crops cost ~11 s per image and 3 points of precision for no F1 gain (tagger-only). The AI
path still uses them (`tag_one_fast`); dropping them there would make every AI image read ~11 s faster, but needs a
full-pipeline benchmark first.

Roadmap status: steps 1-4 done (fast Tagger mode, batch captions, transfer presets, background scan measured).
Remaining: the region editor and the ComfyUI node (later), and the corner-crop decision for the AI path.
