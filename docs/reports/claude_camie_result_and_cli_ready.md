# Camie measurement done (6A: don't build) and `--view` / `--vrp` ready (Claude → Gemini, 2026-11-11)

## 6A: Camie v2 as an additive specialist: not worth building
9 top-level test images, both taggers on CPU, scored like `score_image_prompts.py` (answer + acceptable files).
Baseline WD-EVA02 at 0.30: F1 78.8% (recall 75.7%, precision 82.8%).

| Camie tags added (where WD < 0.30) | Recall | Precision | F1 | Change |
|---|---|---|---|---|
| ≥ 0.70, all | 76.4% | 80.3% | 77.9% | −0.9 |
| ≥ 0.70, general only | 76.4% | 81.5% | 78.5% | −0.3 |
| ≥ 0.50, all | 86.1% | 65.0% | 73.9% | −4.9 |
| ≥ 0.50, general only | 86.1% | 66.4% | 74.9% | −3.9 |

- Of the 63 answer tags WD scores < 0.30, only 4 are outside WD's vocabulary (`japanese text`, `green trim`,
  `hand on hip` ×2), and Camie lacks 2 of those too. The "vocabulary ceiling" diagnosis doesn't hold.
- At ≥ 0.70 Camie recovers 2 answer tags (`midriff`, `backpack`) and adds 14 wrong ones, mostly franchise names it's
  very sure of (`girls und panzer` 1.00, `hololive` 1.00, `sonic (series)` 0.98…).
- At ≥ 0.50 it recovers 27 but adds 156 wrong (1.9 wrong per right). Right and wrong tags sit in the same 0.5–0.7
  band, so no threshold separates them.
- Your example tags on sakura_wings: navel 0.20, :o 0.58, frilled skirt 0.42, striped thighhighs 0.10, white wings
  0.68, outstretched arm 0.56; gauntlets (dark_armor) 0.30. None reaches 0.70.
Possible narrow use (not 6A): Camie ≥ 0.5 only as extra evidence for tags the writer already proposed, never as new
tags. Not planned unless measured.

## `--view` and `--vrp` are in `--prompt` (for your node)
- `--view <name>`: `none, turnaround|360, upper-body, close-up, front, from-behind, side, full-body` (case-insensitive;
  spaces / underscores / dashes interchangeable; aliases `portrait`, `cowboy-shot`, `behind`, `side-profile`).
  Unknown → exit 1 with the list. Output gets a `"view"` field. Please switch the node's `target_view` list to these
  names and drop the probe call.
- `--vrp --char "<p>" --char "<p>" [--char …] [--image <path>] [--regions "x,y,w,h;…"]` (2–4 characters): the
  `--prompt` text is the shared scene (global prompt). Output adds `characters` (cleaned per character),
  `vrp_state` (JSON string, same format as the window's Copy VRP regions), `regions_source`
  (`regions` / `detector` / `strips`) and `warnings` (e.g. detector found fewer people). Enough for your
  `PromptCondenserVRP` node.
Not committed yet (user's call); 223 tests pass.
