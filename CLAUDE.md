# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What it is

`prompt-condenser`: a Rust + Slint desktop app that condenses ComfyUI prompts, fixes camera-framing / body-part conflicts per checkpoint (Anima, Pony V6, Illustrious, SDXL, SD1.5, Flux), and optionally runs a "deep scan" through an LLM served by **Ollama**, a required runtime dependency (`ollama` + `ollama-vulkan` on Arch for GPU). Ollama can run on this PC or another one on the LAN. Not a git repo. The original design doc lives outside the repo, in the Gemini Antigravity brain folder (`implementation_plan.md`).

## Commands

- Build / run: `cargo build`, `cargo run`
- Tests: `cargo test`; single test: `cargo test turnaround_360` (substring match)
- **Arch package (preferred installer)**: `scripts/build_arch_package.sh [--install]` packs the sources into `packaging/arch/` and runs `makepkg`. The result is `packaging/arch/prompt-condenser-<ver>-1-x86_64.pkg.tar.zst`; `--install` also installs it with its dependencies (ollama, ollama-vulkan, vulkan-tools, xdg-desktop-portal, Slint's Wayland/X11/EGL libs). The package ships the binary, a desktop entry, an icon (`packaging/`) and an Ollama systemd drop-in that waits for a Vulkan GPU at boot. `pkgver` is synced from `Cargo.toml`.
- Run with a prompt preloaded: `prompt-condenser "1girl, …"`
- Generic (non-Arch) installer bundle: `scripts/build_installer.sh` → `dist/prompt-condenser-installer{,.tar.gz}` (binary plus `install.sh`; no bundled models). Builds with `-C target-cpu=x86-64-v3` (override via `TARGET_CPU`).
- Install for current user: `./install.sh` (also works from the bundle). It installs and starts Ollama (with an `OLLAMA_VULKAN=1` systemd drop-in), installs the app into `~/.local`, and offers to pull the recommended model (`prompt-condenser-gemma4-12b`) through Ollama. `./install.sh --uninstall` removes the app.

## Architecture

Pipeline: `parser` → `analyzer` → `condenser`, driven from `main.rs`; `ai_engine` is an optional extra pass.

- `parser.rs`: splits the prompt on top-level commas/newlines (respecting `()`, `[]`, `<lora:…>`), unwraps A1111/ComfyUI weight syntax into `PromptToken { weight, had_runaway_weight, original_index, category }`. `TokenCategory`'s declaration order is the output sort order when cleanup is on.
- `profiles.rs`: dropdown enums. **`from_index` values must match the ComboBox order in `ui/app.slint`**. `AiMode` index 2 = AI off, which the UI also hard-codes. `AiModelTier` maps each tier to its Ollama name and its `hf.co/...` pull source.
- `analyzer.rs`: `analyze_with_rules` returns the processed tokens (every suggestion already applied, i.e. mode 0) plus a `PromptChange` list (the cards).
- `condenser.rs`: `build_condensed_output` re-applies each card's `mode` on top of the processed tokens (0 = suggestion, 1 = keep body part and swap framing, 2 = restore original). AI cards (`ChangeAction::AiRewrite`) are opt-in: they start at mode 2 and `apply_ai_card` only edits the prompt once one is accepted. It also emits `highlighted_markdown`, which marks substituted (`PromptToken.replaced`) and added tokens with `<font color><u>`. It's used to list those tags under the After box.
- **The rule-based output never contains tags the user didn't write**, except as direct replacements for one of their tags. Don't reintroduce auto-appended framing/quality/score tags or pinned-tag injection; tests in `analyzer.rs` guard this. New content comes only from the AI, opt-in and highlighted: green = added, amber = substituted (`condenser::Highlight`). The AI Improved prompt is a separate After view (`after_view`).
- `ai_engine.rs`: Ollama HTTP client only; there is no in-process inference.
  - Endpoints: `/api/tags` (model list), `/api/generate` (`format: "json"`, `keep_alive: 30m`; `options.num_gpu: 0` when the mode is CPU only), `/api/ps` (reports GPU vs CPU placement for the status bar).
  - `pull_tier_model`: `/api/pull` of the `hf.co` GGUF, then `/api/copy` under the short name, then `/api/delete` of the long name.
  - Deep scan reply (enforced with a JSON schema in `format`, property order = generation order): `scene_summary`, `scene_breakdown`, `issues`, `improved_prompt` (full rewrite, may add details), `changes` (replace/remove/add/reweight/framing_safe_swap).
  - `convert_ai_changes` turns `changes` into opt-in cards. It drops findings whose `original` isn't in the prompt, that a rule card already covers, or that change nothing. An `original` of `""` means "add". AI card ids start at `AI_CARD_ID_BASE`.
  - Replies that hit the output limit are repaired: `close_truncated_json` keeps everything up to the last complete value, and only fully written changes are kept. Unreadable replies are retried once.
  - Models: `AiModelTier::ALL` offers three uncensored **Gemma 4** tiers ordered lightest to best (`GemmaE2B`, `GemmaE4B`, `Gemma12`), and the app preselects the last installed one. The recommended default is **Maximum Fast — Gemma 4 12B Q4** (`Gemma12`, ~8 GB VRAM). In head-to-head tests (`live_compare`) Gemma 4 12B beat Qwen2.5 7B/14B, Qwen3 14B, and Qwen3.5 9B Q8 on analysis, faithfulness, conflict fixes and negatives, so all Qwen tiers were replaced with Gemma 4 (`GemmaE2B` at **Q8**, ~2.7 GB, and `GemmaE4B` at Q4, ~3.3 GB, for lower-VRAM slots). Q8 mattered for E2B: its Q4 softened explicit prompts. For E4B, Q8 barely helped (5.3 GB), so it stays Q4.
  - Benchmark of the four tiers (4 prompts, BC-250). E2B: 1.8 GB, 8–18 s, decent but softened the explicit prompt. E4B: 3.3 GB, 10–30 s, faithful but adds little. 12B Q4: 8.1 GB, 27–49 s, best overall. 12B Q6: 10.2 GB, 35–60 s, same quality as Q4 but slower. All four are uncensored (no refusals).
  - Higher precision was removed. A Gemma 4 12B Q6 tier ("Maximum Detailed") gave no visible quality gain over Q4 in the benchmark, only ~25–40% slower scans, so it was dropped. **Q8 OOM-crashed the BC-250** (three hard reboots) once Ollama's `OLLAMA_GPU_OVERHEAD` reserve had been lowered. Keep that reserve at 2 GB on shared-memory GPUs.
  - `tier_template()`: returns `None` for all Gemma 4 tiers so they keep their native GGUF template. Scans send `think: false`, and retry without it if a model rejects the setting.
  - Instructions per model family: `ModelFamily::detect(model name)` picks one of Gemma (markdown sections plus a JSON format example from an unrelated astronaut scene), Qwen25 (plain uppercase sections, for user-supplied Ollama models), Qwen3, Small (`<4B`, including `GemmaE2B`: a short version with fewer items) or Generic. The content is shared in `build_ai_system_and_user_prompt_for`; only format, amount and temperature (`ModelFamily::temperature`) differ. `param_billions` reads the size from the name ("32b" is not small).
  - Thinking mode (`AiScanRequest.think`, off in the app) was tested and doesn't work: Ollama rejects `think: true` for the Gemma 4 GGUF ("does not support thinking"), and the retry then silently runs it as a normal scan. Qwen3 (the only other candidate) was removed for poor results.
  - `prompt_guidance()` adds an "ABOUT THIS PROMPT" block to the system prompt, worked out from the Before prompt. It covers scenery vs. multiple characters, LoRAs/trigger words, character names, prose clauses, explicit content, short vs. long prompts, what's already specified vs. missing (via `scene::describe`), and the rule engine's framing conflicts.
  - `talks_to_ollama_api` tests the client against an in-process fake server.
- `tagger.rs`: image → Danbooru tags with SmilingWolf's WD EVA02-Large v3 (ONNX, via the `ort` crate; ONNX Runtime is statically linked and runs on CPU). The model and `selected_tags.csv` download on demand to `~/.local/share/prompt-condenser/wd-eva02-large-tagger-v3/`, and the session is cached in `MODEL`.
  - Preprocessing: pad to a white square, resize to 448, BGR, 0–255, NHWC.
  - `build_prompt`: character tags (threshold 0.85) first, then general tags at the UI threshold. Underscores become spaces (except kaomoji) and parentheses are escaped.
  - Scores are kept in `AppState.last_tags` so the threshold slider re-filters without re-running the model.
  - History: reading prompts saved in PNG metadata, and an Ollama vision-LLM description, were both tried and removed at the user's request.
- Camera angle / pose dropdowns: `profiles::CAMERA_ANGLES` and `POSES` hold (label, tags) pairs, with index 0 = none. A choice replaces the prompt's own angle/pose tags (`group_tags`) in `analyze_with_rules` and shows an `AngleChoice`/`PoseChoice` card. It's also passed to the AI. The "Random" entries are `{a|b|c}` groups.
- Dynamic prompts (`ScanConfig.dynamic`, an Options checkbox): the parser keeps `{…|…}` as one tag, and `tidy_ai_prompt` never splits them. The After view has a third mode, Dynamic (`after_view == 2`): it shows the deep scan's `dynamic_prompt` if there is one, else `dynamic::make_dynamic` (rule-based: one variation group each for camera angle, pose, expression, lighting and time of day, taken from `VARIATION_GROUPS`; identity, outfit and setting are never varied).
- `scene.rs`: live, rule-based scene description (no AI) of the prompt shown in After. Keyword lists sort tags into who/traits/hair/eyes/body/outfit/expression/pose/camera/setting/light/style and produce a headline, a breakdown and a list of what's missing (`render_live_scene` in `main.rs`).
- Recommended negative: `condenser::recommended_negative` combines `CheckpointProfile::default_negative()` with `TargetView::negative_tags()`. `merge_negative` drops anything the positive prompt contains. On the AI Improved view, the deep scan's `recommended_negative` is merged on top (`render_negative` in `main.rs`). Flux gets none.
- `main.rs`:
  - UI state is `Rc<RefCell<AppState>>`, used on the UI thread only.
  - The AI model dropdown is backed by `AppState.model_choices` (the tiers first, then other Ollama models).
  - Blocking work (AI scan, Ollama pulls, connecting, GPU probe) runs on `std::thread`s and reports back with `upgrade_in_event_loop`. Those closures must be `Send`, so they reach the state through the `STATE` thread-local.
  - Edits call `analyze_requested`, which is debounced by 120 ms. `analyze` resets rule-card modes. AI cards survive until the prompt, checkpoint or view changes (`AiKey`) or the AI mode/model changes.
  - Cards live in one persistent `VecModel` that is updated row by row. Don't replace the model on each render, since that rebuilds every card.
- UI size: `Zoom` in `main.rs` dispatches `WindowEvent::ScaleFactorChanged` with monitor scale × `ui_zoom` (default 0.75, shown to the user as 100%; range 70–200% in 10% steps via `ZOOM_BASE`), because Material widgets have fixed minimum sizes. A 250 ms timer re-applies it if the platform resets the scale.
- Dev builds compile dependencies at opt-level 3 (see `Cargo.toml`). Without that, Slint is very laggy.
- `ui/app.slint`: compiled by `build.rs` with the `material-dark` style. Colors come from the `M3` global. Slint `Rectangle` has no `padding`, so put padding on an inner layout.
  - Workspace: `CardsPanel` (a horizontal strip, always on top: `LiveSceneCard`, then the AI `SceneCard`, then the suggestion cards), a `Splitter`, then `PromptColumn`: a left column (Before / After) and a right column (Organized Structure / Recommended Negative). Both columns use `before_ratio` as their vertical stretch so the rows line up; one vertical divider sets `side_width`.
  - Sizes: `cards_strip_height` (preferred height), `before_ratio` (Before vs After stretch share), `side_width` (preferred width of the right column). Use preferred sizes, not fixed sizes bound to a parent's height: the latter causes Slint "Recursion detected" panics.
  - `M3Card` clips its contents.
  - Cards in the strip have a fixed size (scene cards 380 px, suggestion cards 300 px, all strip-high) and scroll their own content with `ScrollBox`. That's a custom wheel handler that rejects Shift + wheel and horizontal scrolls, so the strip's ScrollView still scrolls sideways. Don't put a nested `ScrollView` in a card: it grabs every wheel event. Card buttons are compact `PillButton`s, and a third option wraps to its own row.
  - The After, Recommended Negative and Organized Structure boxes are plain read-only `TextInput`s, so they can be selected and copied. Slint's `StyledText` can't be selected, and layering it with an invisible TextInput didn't work for mouse selection. Colors are therefore not shown inline; `condenser::marked_tags` pulls the substituted and added tags out of `highlighted_markdown` for the lines under After.
  - Clicking the image thumbnail (`remove_image_requested`) clears the loaded image but leaves the prompt alone.
- `main.rs` saves those sizes, `ui_zoom` and the Ollama host to `~/.config/prompt-condenser/settings.json` when the window closes (`Settings`).
