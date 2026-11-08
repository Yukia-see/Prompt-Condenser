# Walkthrough: Dual-Pass Vision & Synergy Rules Integration

The native Rust Dual-Pass Moondream + WD14 + Gemma 4 12B pipeline specified in `ANTIGRAVITY_RUST_MISSION.md` has been integrated into Prompt Condenser. All 1,290 parity tests match the reference Python implementation with zero discrepancies, all existing test suites pass, and the application release binary has been compiled and installed.

---

## 1. Summary of Changes

### Core Rust Dual-Pass Engine
- **[`src/prompt_condenser.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/prompt_condenser.rs)**:
  - Zero-dependency standard-library implementation of the dual-pass architecture.
  - Implements `run_dual_pass`, `WD14DualPassRouter`, `merge_tags`, `load_vocab_from_csv`, Ollama request body generators, and response parsers.
  - Supports Mode A (`SceneShape`) and Mode B (`ScenePose`).
- **[`src/prompt_condenser_parity.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/prompt_condenser_parity.rs)** & **[`tests/parity.json`](file:///home/yukia/Documents/Prompt%20Condensor/tests/parity.json)**:
  - Parity test suite verifying 1,290 test cases against the Python reference (`router`, `analyze`, `merge`, `nollm`, `parse`, `flatten`, `style`, `payload`).
  - Added via `#[cfg(test)]` with zero runtime footprint in release builds.

### Backend Integration & Closures
- **[`src/ai_engine.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine.rs)**:
  - Implemented `run_dual_pass_pipeline` managing the Moondream and Gemma closures.
  - **Moondream**: Passed temperature `0.1` (`MOONDREAM_TEMPERATURE`) with a single retry bump to `0.3` (`MOONDREAM_RETRY_TEMP_BUMP = 0.2`) on empty or refusal replies.
  - **Gemma 4 12B**: Passed temperature `0.35` in JSON format with `num_ctx = 131,072`. Handles vision check with base64 image data when `gemma_vision_pass` is enabled.
  - **Health Check & Fallback**: Queries `GET /api/tags` via `prompt_condenser::ollama_model_pulled`. If Ollama is unreachable or model is absent, falls back to the mechanical tag merge and emits warnings.
- **[`src/danbooru.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/danbooru.rs)**:
  - Added cached vocabulary provider `get_prompt_condenser_vocab()` reading `selected_tags.csv` once and caching in `OnceLock<HashSet<String>>`.
- **[`src/main.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/main.rs)**:
  - Integrated `run_dual_pass_pipeline` into `run_ai_image`, `read_scene_blocking`, and `read_char_blocking`.
  - Added `enable_dual_pass`, `dual_pass_mode_index`, and `use_llm_synthesis` to `Settings` serialization and deserialization.
  - Wired event callbacks and UI updates.

### Slint UI Controls
- **[`ui/app.slint`](file:///home/yukia/Documents/Prompt%20Condensor/ui/app.slint)**:
  - **`[Dual-Pass Vision]`**: FilterChip switch (default: `true`).
  - **`Dual-Pass Mode` Dropdown**: ComboBox selecting between `Scene + Body Shape (Mode A)` and `Scene + Skeletal Pose (Mode B)`.
  - **`[LLM Synthesis]`**: FilterChip toggle for Gemma 4 12B tag synthesis (default: `true`).
  - **`[2nd Pass Visual Check]`**: FilterChip toggle for sending base64 image pixels to Gemma for visual verification (default: `false`, enabled when LLM Synthesis is on).
  - **Pipeline Warning Area**: Dedicated dismissible alert banner displaying `result.warnings` when silent fallbacks occur.

---

## 2. Verification & Validation Results

### Automated Test Suite (`cargo test`)
```bash
cargo test
```
* **Result**: `167 passed; 0 failed; 14 ignored; finished in 0.10s`
* **Parity Test**: `test prompt_condenser_parity::rust_matches_python_reference ... ok` (All 1,290 cases matched 100%).

### Clippy Cleanliness (`cargo clippy --all-targets`)
```bash
cargo clippy --all-targets
```
* **Result**: Succeeded with code 0.

### Release Binary Build & Installation
```bash
cargo build --release
install -m 755 target/release/prompt-condenser ~/.local/bin/prompt-condenser
```
* **Result**: Compiled in 3m 19s and installed release binary to [`~/.local/bin/prompt-condenser`](file:///home/yukia/.local/bin/prompt-condenser) (61 MB).

---

## 3. Checklist of Required Items from `ANTIGRAVITY_RUST_MISSION.md`

| Requirement | Implementation Details |
| :--- | :--- |
| **Files Changed** | `src/prompt_condenser.rs`, `src/prompt_condenser_parity.rs`, `tests/parity.json`, `src/ai_engine.rs`, `src/danbooru.rs`, `src/main.rs`, `src/sceneref.rs`, `ui/app.slint` |
| **Four UI Toggles** | `enable_dual_pass` (FilterChip), `dual_pass_mode` (ComboBox), `use_llm_synthesis` (FilterChip), `gemma_vision_pass` (FilterChip) in Image Reference Toolbar |
| **Moondream Temperature** | `0.1` factual temperature passed in Ollama options; single retry at `0.3` on refusal/empty reply |
| **HTTP Client for Ollama** | `reqwest::blocking::Client` using app's existing client configuration |
| **`selected_tags.csv` Path** | `~/.local/share/prompt-condenser/wd-eva02-large-tagger-v3/selected_tags.csv` |
| **Warnings in UI** | Dismissible notification bar directly under Image Reference Toolbar |
| **Test Verification** | 167/167 tests passed, including `rust_matches_python_reference` (1,290/1,290 cases) |
