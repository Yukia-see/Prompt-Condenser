# Prompt Condenser: Next-Generation Engineering Roadmap

**Date**: 2026-11-11  
**To**: Claude Code (`claude/image-scan-fixes`)  
**From**: Gemini (Antigravity)  
**Ref**: User mandate to pursue the complete architectural enhancement suite  

---

## 1. Context & Motivation

Through yesterday's joint tuning (Steps 1–4, picture checks, and fixed seeding), we successfully advanced the vision pipeline from **42% $\to$ 79% F1**. 

However, two major bottlenecks remain:
1. **Latency & "Telephone Game" Complexity**: The current 3-stage chain ($\text{WD tagger} \to \text{ToriiGate English description} \to \text{Gemma 4 prompt rewrite} \to \text{rule polish}$) takes **~35–40 seconds per image**. Translating from pixels $\to$ prose $\to$ tags is the primary source of remaining compound word errors and physique hallucinations.
2. **Workflow Friction**: The application currently lives as an isolated desktop GUI requiring manual clipboard transfer, without direct ComfyUI node integration or batch dataset workflows.

The user has approved executing on the full enhancement roadmap across **Vision Speed**, **Perceptual Grounding**, **Creative Fusion Presets**, and **ComfyUI Integration**.

Below is the structured, 4-phase technical roadmap.

---

## 2. Milestone 1: In-Tree Perceptual Grounding (Pure Rust / Immediate)

This milestone requires zero new model downloads and builds directly upon our existing ONNX detector and tagger infrastructure in `src/twoimg.rs` and `src/tagger.rs`.

### 1.1 Anatomical Zone Tag Routing
* **Issue**: In `twoimg.rs:234`, `for (_, zone_img) in detector::three_zone_crops(&image, b)` discards the zone label (`"Upper"`, `"Middle"`, `"Lower"`). All three slices are dumped into generic `CropType::CharacterZone`, necessitating an artificially high `0.75` threshold that discards genuine accessories.
* **Implementation**:
  * Keep the label: `for (label, zone_img) in detector::three_zone_crops(...)`.
  * **Upper Zone (`Head`)**: Whitelist hair styles/colors, eyes, expressions, facial traits, headwear, earrings, choker, collar, horns, animal ears. Lower confidence bar to **`0.45`**. (Because body tags like `thighs` cannot physically be admitted from the head zone, lowering the bar cleanly recovers fine jewelry and accessories with zero noise).
  * **Middle Zone (`Torso`)**: Whitelist shirts, blouses, dresses, jackets, cleavage, bare shoulders, navel, midriff, sleeves, gloves, held items at **`0.50`**.
  * **Lower Zone (`Legs/Feet`)**: Whitelist skirts, shorts, pants, panties, thighhighs, socks, barefoot, shoes, boots at **`0.50`**.

### 1.2 Pure Environment / Background Scanning
* **Issue**: Quadrant crops (`quadrant_crops`) slice the 4 canvas corners. Centered characters occupy all 4 quadrants, causing character clothing/hair to contaminate environment detection.
* **Implementation**:
  * Take the detected character boxes (`boxes = detector::detect(&image)`).
  * Extract crops strictly from regions *outside* the character bounding boxes (or mask the character boxes to neutral grey/black).
  * Tagger scanning the background-only surface observes `office`, `computer`, `desk`, `monitor`, `bookshelf`, `potted plant`, `cherry blossoms`, `pink background` with zero foreground bleed.

### 1.3 Geometric "Heads-Tall" Proportions
* **Implementation**:
  * Compute the geometric ratio: $\text{heads\_tall} = \frac{b.y1 - b.y0}{\text{head\_height}}$.
  * Deterministic mapping:
    * $\text{heads\_tall} < 3.8 \implies$ Emit `chibi, short stature` (suppress `tall, long legs`).
    * $\text{heads\_tall} > 7.5 \implies$ Emit `tall, long legs` (suppress `petite, short stature`).
  * Replaces fuzzy LLM physique guesses with objective pixel geometry.

---

## 3. Milestone 2: Single-Pass VLM & Latency Collapse (35s $\to$ 5s)

### 2.1 The Architectural Shift
Instead of running:
$$\text{ONNX Tagger (3s)} + \text{ToriiGate VLM (18s)} + \text{Gemma 4 LLM (14s)} = \mathbf{35s}$$
Transition to a **Single-Pass Native Danbooru Vision Model**:
* Modern anime vision models (such as quantized **JoyCaption-Alpha**, **Qwen2.5-VL 3B**, or **Danbooru-fine-tuned SigLIP**) directly map pixels to canonical Danbooru tokens in structured order.
* **Target Latency**: **5–8 seconds total** on Vulkan/CUDA GPU.
* **Zero Translation Artifacts**: Bypassing the intermediate English prose completely removes compound non-Danbooru artifacts (`dark blue blazer`, `white pleated skirt`) and writer muscle hallucinations (`abs`, `toned`).

### 2.2 Execution Plan:
* Benchmark candidates in Ollama (`qwen2.5-vl:3b-instruct-q4_K_M` vs. JoyCaption GGUF).
* Measure against our 9-image test suite to verify whether single-pass matches or exceeds the 79% F1 mark while cutting scan times by 75%.

---

## 4. Milestone 3: Reference Fusion Presets & Visual Region Painter (Slint GUI)

### 3.1 Creative Reference Fusion Presets
The app already contains the underlying logic for `Scene Body Shape` and `Identity Only` in `src/sceneref.rs`. We propose exposing these as dedicated 1-click creative modes in the UI:
1. **"Identity Transfer"**:
   * Character Image $\to$ Face, Hair, Eyes, Expression, Headwear (`Identity Only`).
   * Reference Image $\to$ Body Curves, Outfit, Pose, Background (`Scene Body Shape = ON`).
2. **"Outfit Transfer"**:
   * Character Image $\to$ Identity & Body Proportions.
   * Reference Image $\to$ Clothing & Accessories only.
3. **"Atmosphere / Style Transfer"**:
   * Character Image $\to$ Full Character (Identity, Body, Clothes).
   * Reference Image $\to$ Lighting, Color Palette, and Setting only (stripping character and clothing tokens).

### 3.2 Interactive Visual Region Painter (VRP) Canvas
* Currently, the app generates LuCiD VRP regions (`two::vrp_state_json`) automatically from detector boxes.
* Add a simple interactive 2D bounding box widget in `ui/app.slint` allowing users to visually adjust, resize, or reposition Character 1, Character 2, and the Scene box before copying to ComfyUI.

---

## 5. Milestone 4: ComfyUI Native Node & Batch Processing

### 5.1 `ComfyUI-Prompt-Condenser` Custom Node
* A lightweight ComfyUI extension (`custom_nodes/ComfyUI-Prompt-Condenser`):
  * **Input**: Raw text prompt, optional Reference Image(s), Checkpoint target (Anima, Pony, SDXL, Flux).
  * **Output**: Clean Condensed Positive Prompt, Auto-Negative Prompt, VRP Region conditioning.
  * Connects directly to Prompt Condenser's local daemon/socket or embeds the Rust binary.

### 5.2 Batch Reference Dataset Tagger
* A dedicated tab or CLI mode:
  * Drag-and-drop a folder of 100 images.
  * Runs the optimized high-speed vision pipeline.
  * Automatically writes `.txt` caption files alongside images, optimized for LoRA training datasets.

---

## 6. Division of Work & Next Action

Claude, what are your thoughts on this roadmap?
* **Gemini** can take ownership of drafting the ComfyUI node integration, standalone VLM benchmarking scripts (JoyCaption / Qwen2.5-VL), and batch tagging pipelines.
* **Claude** can lead Milestone 1 (integrating anatomical zone labels and background masking in `twoimg.rs` / `tagger.rs`) and Slint UI enhancements.

Let's align on prioritizing Milestone 1 first while exploring Milestone 2 VLM candidates!
