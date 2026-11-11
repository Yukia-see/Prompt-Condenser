# 4-Way Multi-Model Benchmark Test Report

**Target Test Image**: [`HQwyhBIbAAAguCD.jpg`](file:///home/yukia/Documents/Prompt%20Condensor/HQwyhBIbAAAguCD.jpg)  
**Subject**: 3 Characters (Pyra, Mythra, Pneuma from Xenoblade Chronicles 2 in swimwear)  
**Test Hardware**: AMD BC-250 (16GB GDDR6 Unified Memory)  
**Execution Date**: November 2026

---

## 1. Benchmark Execution Results

| Metric / Feature | Config 1: 3B $\rightarrow$ 3B | Config 2: 7B $\rightarrow$ 7B | Config 3: 3B $\rightarrow$ 7B | Config 4: 7B $\rightarrow$ 3B |
| :--- | :--- | :--- | :--- | :--- |
| **Pass 1 Vision Model** | Qwen 2.5-VL 3B | Qwen 2.5-VL 7B | Qwen 2.5-VL 3B | Qwen 2.5-VL 7B |
| **Pass 2 Vision Model** | Qwen 2.5-VL 3B | Qwen 2.5-VL 7B | Qwen 2.5-VL 7B | Qwen 2.5-VL 3B |
| **Synthesis LLM** | Gemma 4 12B (Vision Pass) | Gemma 4 12B (Vision Pass) | Gemma 4 12B (Vision Pass) | Gemma 4 12B (Vision Pass) |
| **Execution Time** | **44.98s** | 45.03s | 80.59s | 53.36s |
| **Character Names** | **Retained** (`pyra, mythra, pneuma`) | **Lost** (collapsed to `3girls`) | **Retained** (`pyra, mythra, pneuma`) | **Lost** (collapsed to `3girls`) |
| **Eye Color Accuracy** | Red, Green, Yellow | Red, Green, Yellow | **Red, Green, Yellow, Aqua** | Red, Green, Yellow |
| **Hair Color Accuracy** | Red, Green, Blonde, Aqua | Red, Green, Blonde | **Red, Green, Blonde, Aqua** | Red, Green, Blonde |
| **Wardrobe Details** | White, Black, Yellow bikini | White, Green, Black bikini | White, Black, Green bikini | White, Black, Yellow bikini, Red flower |
| **Schema Leaks / Artifacts**| None | **Leaked** (`expression, face, hair, body shape`) | None | None |
| **Overall Rank** | **BEST (Fast & Accurate)** | **POOR** | **BETTER (Highest Detail, but Slow)** | **GOOD (High Detail, Lost Names)** |

---

## 2. Detailed Output Per Configuration

### Config 1: Symmetrical 3B $\rightarrow$ 3B $\rightarrow$ Gemma 12B (44.98s)
- **Scene Prompt**:
  `white background, simple background, well-lit, soft lighting, straight-on, three-quarter view, symmetrical perspective`
- **Character Prompt**:
  `3girls, multiple girls, pyra, mythra, pneuma, red hair, green hair, blonde hair, long hair, large eyes, expressive eyes, bikini, swimsuit, white bikini, black bikini, yellow bikini, red flower, gold, green, barefoot, thick thighs, wide hips, curvy, long legs, red eyes, green eyes, yellow eyes, aqua hair, swept bangs, ponytail, high ponytail, bob cut, chest jewel, tiara, headpiece, jewelry, gem, strap, strap gap, mismatched bikini, two-tone bikini, skindentation, face, smile, blush, looking at viewer, open mouth, closed mouth, bare shoulders, bare legs, stomach, navel, twisted torso, different hair colors, different eye colors`
- **Analysis**:
  - Extremely fast (44.98s) with no model-swapping latency in VRAM.
  - Retained all three distinct character identities (`pyra, mythra, pneuma`).
  - Correctly differentiated hair and eye colors across the cast.
  - Zero token limit truncation or schema leaking.

---

### Config 2: Symmetrical 7B $\rightarrow$ 7B $\rightarrow$ Gemma 12B (45.03s)
- **Scene Prompt**:
  `white background, bright, evenly lit, frontal view, no depth`
- **Character Prompt**:
  `3girls, red hair, green hair, blonde hair, green eyes, yellow eyes, red eyes, long hair, very long hair, swept bangs, bob cut, high ponytail, ponytail, curvy, toned, thick thighs, wide hips, white bikini, green bikini, black bikini, multicolored bikini, two-tone bikini, mismatched bikini, swimsuit, bikini, jewelry, headpiece, tiara, headband, chest jewel, gem, earrings, barefoot, feet, toes, toenails, skindentation, belly button, navel, stomach, bare shoulders, bare legs, looking at viewer, smile, open mouth, closed mouth, blush, expression, face, hair, body shape`
- **Analysis**:
  - Collapsed individual identities into generic `3girls`.
  - The 7B narrative prose in Pass 1 and Pass 2 caused context bloat, resulting in Gemma dumping structural schema categories (`expression, face, hair, body shape`) at the very end of the prompt when running out of attention capacity.

---

### Config 3: Asymmetric 3B (Pass 1) $\rightarrow$ 7B (Pass 2) $\rightarrow$ Gemma 12B (80.59s)
- **Scene Prompt**:
  `white background, simple background, well-lit, soft lighting, straight-on, three-quarter view, perspective`
- **Character Prompt**:
  `3girls, multiple girls, pyra, mythra, pneuma, anime style, large eyes, red hair, green hair, blonde hair, long hair, very long hair, swept bangs, bob cut, high ponytail, ponytail, red eyes, green eyes, yellow eyes, aqua eyes, aqua hair, expression, smile, blush, open mouth, closed mouth, looking at viewer, bikini, swimsuit, white bikini, black bikini, green bikini, two-tone bikini, multicolored bikini, mismatched bikini, bikini top, bikini bottom, curvy, thick thighs, wide hips, bare shoulders, bare legs, barefoot, toes, toenails, kneepits, headpiece, tiara, chest jewel, gem, jewelry, earrings, strap gap, stomach, navel, twisted torso`
- **Analysis**:
  - **Highest Qualitative Fidelity**: Caught all 3 character names (`pyra, mythra, pneuma`), all 4 hair/eye color pairs (including Pneuma's `aqua eyes, aqua hair`), and fine accessory details (`earrings, tiara, chest jewel, gem, kneepits`).
  - **Downside**: Took 80.59s because Ollama had to load 3B into GPU memory for Pass 1, unload 3B to load 7B for Pass 2, and then unload 7B to load Gemma 12B for synthesis.

---

### Config 4: Asymmetric 7B (Pass 1) $\rightarrow$ 3B (Pass 2) $\rightarrow$ Gemma 12B (53.36s)
- **Scene Prompt**:
  `white background, simple background, bright, evenly lit, natural`
- **Character Prompt**:
  `3girls, multiple girls, red hair, green hair, blonde hair, long hair, swept bangs, red eyes, green eyes, yellow eyes, smiling, looking at viewer, looking to the side, white bikini, black bikini, yellow bikini, multicolored bikini, two-tone bikini, mismatched bikini, bikini, swimsuit, adapted costume, alternate costume, curled hair, bob cut, high ponytail, ponytail, curvy, toned, wide hips, thick thighs, bare shoulders, bare legs, stomach, navel, white bikini bottom, red flower, gold, green leaf, chest jewel, gem, jewelry, tiara, headpiece, earrings, barefoot, feet, toes, toenails, strap gap, twisted torso`
- **Analysis**:
  - In Pass 1, the 7B model generated prose describing the group as a whole without locking down individual character names.
  - Even though 3B executed Pass 2 with itemized checklists, the missing name foundations from Pass 1 resulted in Gemma outputting generic `3girls, multiple girls` instead of `pyra, mythra, pneuma`.
  - Caught interesting micro-garments (`red flower, gold, green leaf, adapted costume`), but lost the core character identity.

---

## 3. Final Rating & Recommendation

1. **BEST**: **Config 1 (3B $\rightarrow$ 3B $\rightarrow$ Gemma 12B)**
   - **Why**: Fastest runtime (~45s), zero model swapping latency, retained all character names, zero hallucinations, perfectly balanced tags. Recommended default for daily production use.
2. **BETTER**: **Config 3 (3B $\rightarrow$ 7B $\rightarrow$ Gemma 12B)**
   - **Why**: Highest detail ceiling (detected Pneuma's aqua eyes/hair and micro-jewelry), but incurs an ~80s runtime penalty due to consecutive VRAM swapping between 3B, 7B, and 12B.
3. **GOOD**: **Config 4 (7B $\rightarrow$ 3B $\rightarrow$ Gemma 12B)**
   - **Why**: Faster than Config 3, but loses individual character names due to 7B's broad Pass 1 narrative.
4. **POOR**: **Config 2 (7B $\rightarrow$ 7B $\rightarrow$ Gemma 12B)**
   - **Why**: Loses character names and suffers from schema category tag leaks.
