# Vision model test: Gemma 4 12B vs ToriiGate (Claude → Gemini, 2026-11-10)

User request: would Gemma 4 12B read images better than ToriiGate? Tested on one image (user: only one), the weakest:
`tests/eval_local/sakura_wings.jpg`, each variant twice, current code (`d592959`), same answer file and scorer.
Raw outputs: `tests/eval_local/results/vision_v1_torii.txt`, `vision_v2_12b_reads.txt`, `vision_v3_12b_both.txt`.

| Variant | F1 (2 runs) | Time per read |
|---|---|---|
| ToriiGate reads, E4B writes (current) | **40% / 40%** | 31-39 s |
| Gemma 4 12B reads (checklist form), E4B writes | 30% / 28% | 48-63 s |
| Gemma 4 12B reads and writes in one call | 36% / 36% | 32 s (2nd: 4 s, cached) |

- ToriiGate: most detail, nearly all right (character, navy sailor top with red/white trim, white pleated miniskirt,
  beret, wings, reaching arm with foreshortening, peace sign, antenna hair); missed the striped thighhighs.
- 12B as reader: invented twin tails, long hair, rabbit ears, a heart tail, a hood, pale skin, boots; only better
  detail: "thigh-high socks with black bands".
- 12B alone: one generic sentence; the prompt is mostly tagger tags plus filler (sparkling eyes, soft lighting).
Conclusion: keep ToriiGate as the image reader; the 12B is a better writer than reader and twice the memory.

Side finding: `tomoeda elementary school uniform` (0.96), `ooarai military uniform` (0.89) and `w` (0.65) still pass
in every variant: the tagger is sure of them, so the ≥ 0.50 recovery threshold doesn't stop them. Candidate rule: a
named uniform / costume tag (`<name> uniform`) that the description contradicts or doesn't mention is dropped.

On your next tasks: two-tier scoring yes (with `invisible` decided the same way on both charts); writer filler Rule 1
(meta words) yes, Rule 2 (drop every unknown phrase the tagger doesn't back) no: it removes correct descriptions like
"pink heart chest piece" and "navy blue sailor top", and dropping unknown phrases once emptied a prompt.
