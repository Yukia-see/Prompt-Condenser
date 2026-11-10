# Changelog and measurements

History moved out of CLAUDE.md: benchmarks, and models or modes that were tried and dropped.

## Models and Deep Scan

- `live_audit` on `Best` (Gemma 4 12B abliterated, Anima/Pony/Flux with 1-3 characters, 2026-11-08): all checks pass except Pony 3 (no cards survived after 2 re-scans, 55 s). The model sometimes writes placeholder text such as `{missing framing tag placeholder}` into the negative prompt. `enforce_checkpoint_rules` (run on every scan reply) drops a Flux negative and puts a checkpoint's missing quality tags first (`ensure_quality_tags`, AI text only; guarded by `rules_a_small_model_forgets_are_enforced`). After it, `Good` (E4B, 10-30 s per scan) passes the whole audit and `Better` (Llama 3.1 8B, 9-29 s) fails only "shared scene kept apart" on Pony 2 and 3 (it mixes the shared scene into the characters). Prefer `Good` or `Best` until Llama's multi-character instructions are tuned.
- Historical benchmark of the previous lineup (Gemma 4 E2B/E4B/12B builds, 4 prompts, BC-250). E2B: 1.8 GB, 8–18 s, decent but softened the explicit prompt. E4B: 3.3 GB, 10–30 s, faithful but adds little. 12B Q4: 8.1 GB, 27–49 s, best overall. 12B Q6: 10.2 GB, 35–60 s, same quality as Q4 but slower. All four are uncensored (no refusals).
- Higher precision was removed. A Gemma 4 12B Q6 tier ("Maximum Detailed") gave no visible quality gain over Q4 in the benchmark, only ~25–40% slower scans, so it was dropped. **Q8 OOM-crashed the BC-250** (three hard reboots) once Ollama's `OLLAMA_GPU_OVERHEAD` reserve had been lowered. Keep that reserve at 2 GB on shared-memory GPUs.
- Thinking mode (`AiScanRequest.think`, off in the app) was tested and doesn't work: Ollama rejects `think: true` for the Gemma 4 GGUF ("does not support thinking"), and the retry then silently runs it as a normal scan. Qwen3 (the only other candidate) was removed for poor results.

## Image reads

- Local test set baseline (2026-11-09, 6 images in tests/eval_local, default chain tagger → ToriiGate → E4B → polish, `tools/eval_images.sh`): average F1 41% (gym girl 59, raccoon girl 47, pencil sketch 43, two gym girls 39, Rouge 35, two-panel schoolgirl 26), ~30 s per image. Weak spots: the sketch was called a "humanoid robot" with black hair; two-panel images are read as one girl.
- Bigger local test set (19 images, 2026-11-09): baseline 31% average F1 (recall 33, precision 30). Sketch / panel / name / filler fixes: 39% (recall 41, precision 38); biggest gains on the three schoolgirls (8 → 32), two-panel schoolgirl (21 → 43), office pair (28 → 49), Rouge (27 → 46), raccoon girl (42 → 59). One image swings about ±5 points between runs. The logo fell 26 → 11 in that run because the writer invented a girl; the `no humans` guard was added after.
- MiniCPM-V 2.6 (8B) as the vision model, raccoon image: 38% vs ToriiGate 42-47%, same speed (37 s); long hair, cat ears and thighhighs were wrong, glasses and red eyes missing. Removed.
- Hardening fixes (2026-11-10, joint Gemini / Claude review): 19-image score 39% → 41% (recall 44, precision 40). The logo recovered (11 → 26) with the `no humans` guard; two blondes 27 → 48, three schoolgirls 32 → 45, beach 38 → 50. Drops (chibi, two-panel schoolgirl, Rouge, -8 to -12) are writer wording differences between runs, not tags removed by the new rules.
