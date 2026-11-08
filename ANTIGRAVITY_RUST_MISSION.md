# MISSION CONTEXT FOR ANTIGRAVITY (Rust / Slint app)

We are adding the `Prompt-Condenser` dual-pass pipeline to the Rust desktop app (`src/main.rs`, `src/ai_engine.rs`, Slint UI). A finished, tested Rust implementation is provided. **Your job is to integrate it, not to redesign or rewrite its logic.**

## 1. What is provided (`prompt_condenser_rust.zip`)

| Path | What it is |
|---|---|
| `src/prompt_condenser.rs` | The whole pipeline. Standard library only, no new crates. Drop-in module. |
| `src/prompt_condenser_parity.rs` | Test-only module: checks the Rust output equals the Python reference on 1,290 golden cases. |
| `tests/parity.json` | The golden cases, generated from the Python reference. Keep next to `Cargo.toml`. |
| `reference/prompt_condenser_dual_pass.py` | The Python reference implementation (source of truth for behavior and data). |
| `tools/build.sh` + `tools/*` | Regenerates the data tables in `prompt_condenser.rs` and `parity.json` from the Python reference. |

Verified before handoff: `cargo test` 34/34 (including parity), `cargo clippy --all-targets` clean, and the module builds and runs inside a plain binary crate. Do not edit the generated data block in `prompt_condenser.rs` by hand; change the Python reference and run `tools/build.sh`.

## 2. What the pipeline does

1. WD14 tags the whole image (Danbooru-style tags, one comma-separated string). Moondream describes the same image twice with two different prompts (prose).
2. Moondream prose is converted to WD-style tags (longest match against a tag vocabulary).
3. WD14 adjusts the Moondream tags. **WD14 wins on attributes** (colors, hair length, breast size). **Moondream wins on pose and camera angle**. Negated mentions ("no glasses") are dropped; hedged mentions ("appears to be") are kept only if WD14 agrees.
4. The tags are split into `scene_prompt` and `character_prompt` by mode.
5. Optionally Gemma (`huihui_ai/gemma-4-abliterated:12b` through Ollama, 128K context) polishes the merged tags, optionally also looking at the image (vision pass).

Modes (dropdown, `Mode::parse("scene_shape" | "scene_pose")`):
* `scene_shape`: Pass 1 = Scene + Body Shape, Pass 2 = Face + Clothes.
* `scene_pose`: Pass 1 = Scene + Skeletal Pose, Pass 2 = Character + Body Shape.

## 3. The API you will call

```rust
let result: PipelineResult = run_dual_pass(
    &wd14_tags,            // &str, e.g. "1girl, solo, blue_dress, sitting"
    &PipelineOptions { mode, use_llm_synthesis, gemma_vision_pass },
    &Config::default(),    // quality prefix, paren escaping, retries, token caps
    Some(&vocab),          // HashSet<String> from load_vocab_from_csv(selected_tags.csv text)
    &moondream_closure,    // Fn(&str prompt, f32 temperature) -> Result<String, String>
    gemma_closure_or_none, // Option<&dyn Fn(&GemmaRequest) -> Result<GemmaReply, String>>
);
// result.scene_prompt, result.character_prompt, result.warnings
```

* `run_dual_pass` never fails and never panics on model errors. Every failure (Moondream error or refusal, Ollama down, Gemma error, unusable Gemma JSON) falls back and adds a message to `result.warnings`.
* It is **blocking**. Call it from a worker thread (or `spawn_blocking` if the app is async), never from the UI thread.
* The single-pass path is not part of the module: when the dual-pass toggle is off, keep calling the existing code.

Helpers for the Ollama transport (you provide the HTTP call with whatever client the app already uses):
* `ollama_chat_request_body(model, text, temperature, num_ctx, num_predict, image_b64, json_mode) -> String` builds the `POST /api/chat` body.
* `parse_ollama_chat_response(body) -> Result<GemmaReply, String>` parses the reply (`done_reason == "length"` triggers the module's one retry with a higher token cap).
* `ollama_model_pulled(tags_body, model) -> Result<bool, String>` parses `GET /api/tags`.

## 4. Integration tasks

1. **Add the files.** Copy `src/prompt_condenser.rs` and `src/prompt_condenser_parity.rs` into `src/`, and `tests/parity.json` next to `Cargo.toml`. In `main.rs` (or `lib.rs`) add:
   ```rust
   #[allow(dead_code)] // until every helper is wired up
   mod prompt_condenser;
   #[cfg(test)]
   mod prompt_condenser_parity;
   ```
   Do not add dependencies for this module.
2. **Moondream closure.** Wrap the app's existing Moondream call as `Fn(&str, f32) -> Result<String, String>` (prompt, temperature). The closure owns/captures the image. Check whether the Moondream runtime in use accepts a temperature; if it does not, use what it supports and leave a comment. Return `Err(message)` for any failure rather than panicking.
3. **WD14 string and thresholds.** The module only receives the tag string. Pass `WD14_GENERAL_THRESHOLD` (0.22) and `WD14_CHARACTER_THRESHOLD` (0.85) to the existing WD14 tagger.
4. **Vocabulary.** Find the `selected_tags.csv` that ships with the WD14 model in use, read it once at startup (or first use), call `load_vocab_from_csv`, and pass the set as `vocab`. Without it, conversion only recognizes a small built-in list.
5. **Gemma via Ollama.** Build the Gemma closure from the three Ollama helpers using the app's existing HTTP client (check `Cargo.toml` first: `reqwest`, `ureq`, or whatever is already there). Use `GEMMA_OLLAMA_MODEL` and `GEMMA_CONTEXT_WINDOW` (131072). `GemmaRequest.with_image` is true on the vision pass: then base64-encode the image and pass it as `image_b64`; otherwise pass `None`. Use JSON mode (`json_mode = true`). Respect `$OLLAMA_HOST`, default `http://localhost:11434`.
6. **Health check.** Before the first Gemma call (or when the toggle is switched on), call `GET /api/tags` and `ollama_model_pulled`. If Ollama is unreachable or the model is missing, pass `None` as the Gemma closure; the module then uses the mechanical merge and records a warning. For a missing model, also show `ollama_not_pulled_message(model)`.
7. **Context window.** The model supports 128K, but the real input is only about 1.5K tokens. If the GPU runs out of memory or Ollama offloads to CPU, lower `GEMMA_CONTEXT_WINDOW` (8192 or 16384). That is the only change needed.
8. **UI wiring (Slint).** Add these controls with these defaults:
   * `enable_dual_pass` (bool, default on): off means call the existing single-pass code and ignore the rest.
   * `dual_pass_mode` (dropdown: `scene_shape`, `scene_pose`; default `scene_shape`).
   * `use_llm_synthesis` (bool, default on).
   * `gemma_vision_pass` (bool, default off): only enabled when `use_llm_synthesis` is on. Needs a multimodal Gemma model (the 12b tag accepts images).
9. **Show `warnings`.** Display `result.warnings` in the UI (a small notice area or a tooltip) whenever the list is not empty, so a silent fallback is never mistaken for a normal result.
10. **Run the tests.** `cargo test` must pass, including `prompt_condenser_parity`. Run it again after your integration changes.

## 5. Rules

* Do not rewrite or "improve" the pipeline logic, prompts, keyword sets or reconcile rules. If something looks wrong, say so in your report instead.
* If you want different behavior, change `reference/prompt_condenser_dual_pass.py` and its tests first, run `tools/build.sh`, and make sure `cargo test` and `python3 reference/test_prompt_condenser.py` both pass. Rust and Python must stay identical.
* Do not hardcode `1girl, solo`; subject tags come from WD14.
* `Config::quality_prefix` is configurable (empty string disables it). `Config::escape_parentheses` (default off) writes `(` as `\(` for A1111/ComfyUI prompt syntax; expose it only if the downstream consumer needs it.
* Do not log or store the user's images.

## 6. Out of scope (not in this module, do not invent)

* The 7-crop region scan (full image, face, upper body, middle, lower body, hands, held items, scene-only inverse). Both Moondream passes currently see the whole image.
* Any "directive" rules from other specs (prop preservation, wardrobe completeness / missing bottoms) and weighted-tag handling (`(tag:1.1)` for Anima checkpoints). They are not part of this pipeline. If the app needs them, ask for the spec instead of guessing.
* Moondream may ignore "ignore X" instructions, so some content leaks between passes. This is inherent to the model.

## 7. Report back

When done, report: files changed, how the four toggles are exposed, how the Moondream temperature is passed, the HTTP client used for Ollama, the `selected_tags.csv` path found, where `warnings` appear in the UI, and the `cargo test` result.
