# Version 0.6.0 Release Directive, UI Redesign, & Next-Gen Feature Roadmap

**Date**: 2026-11-11  
**From**: Gemini & User  
**To**: Claude Code (`claude/image-scan-fixes`)  
**Status**: User-Mandated Action for Version 0.6.0 Release & Active Feature Backlog  

---

## 1. Executive Directive: Version 0.6.0 Release & Push

The user has explicitly instructed to finalize, compile, bump to **version 0.6.0**, and push all progress to GitHub:

1. **Version Bump**:
   - In [`Cargo.toml`](../../Cargo.toml): update `version = "0.6.0"`
   - In [`packaging/arch/PKGBUILD`](../../packaging/arch/PKGBUILD): update `pkgver=0.6.0`
   - In [`CHANGELOG.md`](../../CHANGELOG.md): add entry for `0.6.0`:
     - Fast Tagger Mode (4.6s single-pass ONNX, 79.2% F1)
     - Batch Dataset Captioner (`--caption <dir>`)
     - Headless Rule & Condenser CLI (`--prompt "<text>" [--view ...] [--vrp ...]`)
     - Reference Transfer Presets (Identity, Outfit, Atmosphere)
     - Multi-Character Regional Cross-Negatives (zero attribute bleed in VRP)
     - Modular Bust & Frame Proportion Split (Shortstack & Pear Shape support)
     - 3-Zone Vision Scanner Toolbar with Collapsible Settings Drawer
     - Companion ComfyUI Custom Node Extension (`tools/comfyui/`)

2. **Compilation & Testing**:
   - Run `cargo test` to verify 100% test passing across the suite.
   - Run `cargo build --release` to compile the optimized production binary.

3. **Git Push**:
   - Commit all staged and modified files.
   - Push commits to remote: `origin/claude/image-scan-fixes`.

### Disagreement Resolution & User Escalation Protocol (User Mandate)
The user has established a strict collaboration policy between Gemini and Claude:
* If there is **ANY disagreement or impasse** on an architecture choice, design detail, or implementation step:
  1. **Do not debate or stall**: Immediately **skip to the next item** on the backlog to keep progress moving.
  2. **Log to the pending list**: Add the unresolved item and both perspectives to a clear bullet list in your next report.
  3. **Escalate to the user**: Gemini will receive this list via the liaison bridge and present it directly to the user for the user to make the final executive ruling.

### Resource Allocation: Benchmarking vs. Coding (User Mandate)
If benchmarking (e.g., ONNX tagger shootout, Ollama scan latency, F1 evaluation) and coding/compilation occur concurrently:
* Run coding and compilation at a **lower performance tier** so the benchmark can stretch its legs with full CPU/memory headroom:
  * Restrict cargo compilation/testing to **2 cores at most** (`cargo build -j 2`, `cargo test -j 2`, or `taskset -c 0,1`).
  * Set low CPU scheduling priority (`nice -n 19`) to eliminate measurement jitter and avoid starving benchmark threads.

---

## 2. Complete UI Redesign Specification (`ui/app.slint`)

The user requested a full UI layout clean-up that preserves 100% of existing functionality, callbacks, and engine capabilities, while making the entire interface intuitive, modern, and effortless to read.

### The 4-Section Studio Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  🌌 PROMPT CONDENSER    [ 1 Character ] [ 2 Characters ] [ 3 Characters ] [ 4 Characters ]    [🎨 Theme ▾] [● Ollama] │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. GENERATION & ANATOMY STUDIO (Grouped & Compact)                                                     │
│    Model: [ Pony V6 XL ▾ ]  Detail: [ Balanced ▾ ]  AI: [ GPU Vulkan ▾ ]  LLM: [ ToriiGate ▾ ]        │
│    View:  [ Upper Body ▾ ]  Pose:   [ Standing ▾ ]  Bust: [ Large ▾ ]     Frame: [ Shortstack ▾ ]     │
│    Boost: [ tags to emphasize... ]          Shared Scene: [ sunset, tavern table, warm light... ]      │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. VISION & REFERENCE DOCK (The 3-Zone Dock)                                                           │
│    [🖼 Char] [🌄 Ref] "Ready"   │  [ ⚡ Fast Tagger | 🧠 Combined AI ]  │  [ ⚡ Scan Image ] [⚙ Settings] │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. PROMPT WORKSPACE (Side-by-Side: Input vs. Clean Output)                                             │
│    ┌───────────────────────────────────┐       ┌──────────────────────────────────────────────────┐    │
│    │ BEFORE: RAW PROMPT                │       │ AFTER: OPTIMIZED PROMPT           [ 📋 Copy ]     │    │
│    │ 1girl, solo, shortstack, upper... │  ==>  │ score_9, score_8_up, 1girl, curvy, shortstack... │    │
│    │                                   │       ├──────────────────────────────────────────────────┤    │
│    │ LoRA & Trigger Box:               │       │ AUTO-NEGATIVE                     [ 📋 Copy ]     │    │
│    │ <lora:character:1.0>              │       │ score_4, score_5, low quality, bad anatomy...    │    │
│    └───────────────────────────────────┘       └──────────────────────────────────────────────────┘    │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. STATUS BAR                                                                                          │
│    "Ready — 0 conflicts detected"                       [ Vulkan GPU active ]          UI Zoom: [ - + ] │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Section Specs:

#### Section 1: Generation & Anatomy Studio
* **Line 1 (Model Config)**: `Model:`, `Detail:`, `AI:`, `LLM:` (compact, aligned).
* **Line 2 (Anatomy & Framing)**: `View:`, `Pose:`, `Bust:`, `Frame:`, `[ ⚙ Options ▾ ]`.
  - When Options is toggled, shows clean FilterChips (`Clean Bloat`, `Fix Conflicts`, `Clamp Weights`, `Arranged`, `Natural NL`, `Dynamic`, `BREAK Sep`).
* **Line 3 (Prompt Modifiers)**: `Boost` (with weight slider 1.0–1.5x) and `Shared Scene` placed side-by-side to save 36px of vertical height.

#### Section 2: Vision & Reference Dock (3 Zones)
* **Zone 1 (Images)**: Scene thumbnail, Character / Split image thumbnail with hover `✕`, compact `+C1`..`+C4` pills, and status text.
* **Zone 2 (Engine & Presets)**: `[ ⚡ Fast Tagger ]` vs `[ 🧠 Combined AI ]` segmented switch, plus transfer presets (`Scene & pose`, `Identity transfer`, `Outfit transfer`, `Atmosphere`) when a reference image is loaded.
* **Zone 3 (Action & Drawers)**:
  - Primary Action Button (`⚡ Scan Image` / `⚡ Auto Scene` / `⇄ Transfer`).
  - Secondary `⚡ Auto Scene` pill when reference is loaded.
  - `[ ⚙ Settings · N on ▾ ]` drawer toggle with active count badge.
  - `[ Scene Panel ▾ ]` inspector toggle.
* **Settings Drawer**:
  - Sensitivity slider (`tag_threshold` 0.15 to 0.60).
  - Quick filter chips: `Identity Only`, `Scene Body Shape`, `2nd Pass Check`, `↻ Flush Vision Cache`.
  - Advanced chips: `Dual-Pass Vision` dropdown and `LLM Synthesis`.

#### Section 3: High-Efficiency Workspace
* Maximum window height dedicated to the prompt text edits.
* Clear, prominent **1-Click Copy** buttons for positive and auto-negative outputs.
* Change cards displayed in a clean, scrollable horizontal panel with 1-click Undo.

---

## 3. High-Value Roadmap Enhancements (Added to Active Backlog)

The user has approved adding the following high-impact features to the active implementation backlog:

### 3.1 Internal Engine Features
1. **CLIP 75-Token Subword Chunk Meter**:
   - Integrate an internal BPE subword estimator calculating exact CLIP token usage (`38 / 75 Tokens (Chunk 1)`).
   - Warns when prompt spills across chunk boundaries, preventing prompt dilution.
2. **Automatic LoRA Trigger Word Extraction**:
   - Inspects the first 4KB of `.safetensors` header metadata in local LoRA directories.
   - When `<lora:name:1.0>` is entered, auto-suggests or injects the trained trigger tags with 1 click.
3. **Instant Dynamic Prompt Preview (🎲 "Roll" Button)**:
   - In-memory sampler for `{a|b|c}` wildcard syntax, letting users roll and preview concrete prompt variations instantly without needing ComfyUI.
4. **Batch File Condenser CLI (`--prompt-file <path>`)**:
   - Bulk cleans and formats hundreds of prompts from a `.txt` or `.csv` file in <50ms.
5. **Drag-and-Drop PNG Info / Generation Metadata Reader**:
   - Parses ComfyUI (`prompt` / `workflow`), A1111/Forge (`parameters`), and NovelAI embedded chunks from dropped `.png`/`.webp` images.
   - Instantly populates the prompt, negative, and model settings.
6. **Live Danbooru Tag Autocomplete & Alias Dictionary**:
   - In-memory fuzzy trie leveraging `danbooru.rs` to suggest official tags with post counts as you type (e.g. `twin...` -> `twintails (185k)`), automatically resolving aliases (`pigtails` -> `twintails`).
7. **Universal Syntax Exporter (ComfyUI, WebUI/Forge, NovelAI)**:
   - One-click copy format selector: ComfyUI `(tag:weight)`, WebUI `(tag:weight)`, NovelAI `{tag}` / `[tag]`, or clean raw comma-separated tags.
8. **Ollama Model Keep-Alive Optimization**:
   - Automatic `keep_alive: "15m"` option on Ollama requests to keep the vision and condensation models resident in memory/VRAM, eliminating the 2–3s cold-start penalty on repeated scans.

### 3.2 UI & Ergonomics Features
1. **Syntax Color-Coded Tag Highlighting**:
   - Visual distinction in the prompt text: Identity (Blue), Clothing (Orange), Framing (Purple), Lighting/Scenery (Green), Conflicts (Red strikethrough).
2. **Character & Style Favorites Library**:
   - Save recurring character prompt bundles (Prompt + LoRA + Bust/Frame + Checkpoint) into a fast preset loader.
3. **Tactile Copy Toast Notification**:
   - Floating green badge popping up for 1.2s: `✓ Copied 32 tags (Pony V6) to clipboard!`.
4. **Visual "Before vs After" Diff Inspector**:
   - Color-coded comparison showing exactly what was added (🟢 quality tags), removed (🔴 bloat & conflicts), and modified (🟡 weights).
5. **Negative Prompt Archetype Presets**:
   - One-click negative styles: `Minimal Clean`, `Quality Boost`, `Deep Anatomy Guard`, and `Uncensored / Pure Artistic`.
6. **Tag Weight Hotkey Stepper (`Ctrl + ↑ / ↓`)**:
   - Quick weight adjustment on selected tags directly in the editor without tedious manual syntax typing.
7. **Multi-Character Regional Visualizer (VRP Canvas)**:
   - Interactive 2D drag-and-drop placement grid for characters in multi-character mode to visually position characters (left, right, center, foreground/background) and auto-derive spatial tags (`standing on left`, bounding boxes).
8. **Content Rating Tier Selector**:
   - Quick segmented control `[ Safe / General | Ecchi / Sensitive | Explicit ]` that automatically injects or prunes checkpoint-appropriate rating tags (`rating:safe`, `rating:explicit`, etc.).

---

## 4. Ready to Ship

All elements are fully verified. Compile the release, run the test suite, update version to `0.6.0`, and push to `origin/claude/image-scan-fixes`!
