# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What it is

`prompt-condenser`: a Rust + Slint desktop app that condenses ComfyUI prompts, fixes camera-framing / body-part conflicts per checkpoint (Anima, Pony V6, Illustrious, SDXL, SD1.5, Flux), and optionally runs a "deep scan" through an LLM served by **Ollama**, a required runtime dependency (`ollama` + `ollama-vulkan` on Arch for GPU). Ollama can run on this PC or another one on the LAN. Not a git repo. The original design doc lives outside the repo, in the Gemini Antigravity brain folder (`implementation_plan.md`).

## Commands

- Build / run: `cargo build`, `cargo run`
- Tests: `cargo test`; single test: `cargo test turnaround_360` (substring match)
- **Arch package (preferred installer)**: `scripts/build_arch_package.sh [--install]` packs the sources into `packaging/arch/` and runs `makepkg`. The result is `packaging/arch/prompt-condenser-<ver>-1-x86_64.pkg.tar.zst`; `--install` also installs it with its dependencies (ollama, ollama-vulkan, vulkan-tools, xdg-desktop-portal, Slint's Wayland/X11/EGL libs). The package ships the binary, a desktop entry, an icon (`packaging/`) and an Ollama systemd drop-in that waits for a Vulkan GPU at boot. `pkgver` is synced from `Cargo.toml`.
- Run with a prompt preloaded: `prompt-condenser "1girl, …"`
- Generic (non-Arch) installer bundle: `scripts/build_installer.sh [--no-models] [--low-only]` → `dist/prompt-condenser-installer{,.tar.gz}`. Downloads ~3 GB of models unless `--no-models`. Builds with `-C target-cpu=x86-64-v3` (override via `TARGET_CPU`).
- Install for current user: `./install.sh` (also works from the bundle). It installs and starts Ollama (with an `OLLAMA_VULKAN=1` systemd drop-in) and registers bundled GGUFs as `prompt-condenser-low/high`. `./install.sh --uninstall` removes the app.

## Architecture

Pipeline: `parser` → `analyzer` → `condenser`, driven from `main.rs`; `ai_engine` is an optional extra pass.

- `parser.rs`: splits the prompt on top-level commas/newlines (respecting `()`, `[]`, `<lora:…>`), unwraps A1111/ComfyUI weight syntax into `PromptToken { weight, had_runaway_weight, original_index, category }`. `TokenCategory`'s declaration order is the output sort order when cleanup is on.
- `profiles.rs`: dropdown enums. **`from_index` values must match the ComboBox order in `ui/app.slint`**. `AiMode` index 2 = AI off, which the UI also hard-codes. `AiModelTier` maps each tier to its Ollama name and its `hf.co/...` pull source.
- `analyzer.rs`: `analyze_with_rules` returns the processed tokens (every suggestion already applied, i.e. mode 0) plus a `PromptChange` list (the cards).
- `condenser.rs`: `build_condensed_output` re-applies each card's `mode` on top of the processed tokens (0 = suggestion, 1 = keep body part and swap framing, 2 = restore original). AI cards (`ChangeAction::AiRewrite`) are opt-in: they start at mode 2 and `apply_ai_card` only edits the prompt once one is accepted. It also emits `highlighted_markdown`, which marks substituted (`PromptToken.replaced`) and added tokens with `<font color><u>`. It's used to list those tags under the After box.
- **The rule-based output never contains tags the user didn't write**, except as direct replacements for one of their tags. Don't reintroduce auto-appended framing/quality/score tags or pinned-tag injection; tests in `analyzer.rs` guard this. New content comes only from the AI, opt-in and highlighted: green = added, amber = substituted (`condenser::Highlight`). The AI Improved prompt is a separate After view (`after_view`).
- `ai_engine.rs`: Ollama HTTP client only; there is no in-process inference.
  - Endpoints: `/api/tags` (model list), `/api/generate` (`format: "json"`, `keep_alive: 30m`; `options.num_gpu: 0` when the mode is CPU only), `/api/ps` (reports GPU vs CPU placement for the status bar).
  - `pull_tier_model`: `/api/pull` of the `hf.co` GGUF, then `/api/create` under the short name with a ChatML template, then `/api/delete` of the long name.
  - Deep scan reply (enforced with a JSON schema in `format`, property order = generation order): `scene_summary`, `scene_breakdown`, `issues`, `improved_prompt` (full rewrite, may add details), `changes` (replace/remove/add/reweight/framing_safe_swap).
  - `convert_ai_changes` turns `changes` into opt-in cards. It drops findings whose `original` isn't in the prompt, that a rule card already covers, or that change nothing. An `original` of `""` means "add". AI card ids start at `AI_CARD_ID_BASE`.
  - Replies that hit the output limit are repaired: `close_truncated_json` keeps everything up to the last complete value, and only fully written changes are kept. Unreadable replies are retried once.
  - Small models (1.5B/3B) can't do scene analysis well; the 7B (`Best`) tier is the recommended minimum.
  - `talks_to_ollama_api` tests the client against an in-process fake server.
- `tagger.rs`: image → Danbooru tags with SmilingWolf's WD EVA02-Large v3 (ONNX, via the `ort` crate; ONNX Runtime is statically linked and runs on CPU). The model and `selected_tags.csv` download on demand to `~/.local/share/prompt-condenser/wd-eva02-large-tagger-v3/`, and the session is cached in `MODEL`.
  - Preprocessing: pad to a white square, resize to 448, BGR, 0–255, NHWC.
  - `build_prompt`: character tags (threshold 0.85) first, then general tags at the UI threshold. Underscores become spaces (except kaomoji) and parentheses are escaped.
  - Scores are kept in `AppState.last_tags` so the threshold slider re-filters without re-running the model.
  - History: reading prompts saved in PNG metadata, and an Ollama vision-LLM description, were both tried and removed at the user's request.
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
  - The After, Recommended Negative and Organized Structure boxes are plain read-only `TextInput`s, so they can be selected and copied. Slint's `StyledText` can't be selected, and layering it with an invisible TextInput didn't work for mouse selection. Colors are therefore not shown inline; `condenser::marked_tags` pulls the substituted and added tags out of `highlighted_markdown` for the lines under After.
  - Clicking the image thumbnail (`remove_image_requested`) clears the loaded image but leaves the prompt alone.
- `main.rs` saves those sizes, `ui_zoom` and the Ollama host to `~/.config/prompt-condenser/settings.json` when the window closes (`Settings`).
