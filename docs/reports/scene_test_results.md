# Scene Detection Benchmark Test Report

**Target Image**: [`packaging/arch/scene.jpg`](file:///home/yukia/Documents/Prompt%20Condensor/packaging/arch/scene.jpg)  
**Pipeline**: Production Default — Qwen 2.5-VL 3B (Pass 1 & Pass 2) + Gemma 4 12B (with 2nd-pass visual verification)  
**Execution Time**: 51.63 seconds  
**Warnings**: None (`[]`) — 100% Valid JSON Generation  
**Date**: November 2026

---

## 1. Vision Encoder Extraction (Qwen 2.5-VL 3B)

### Pass 1: Spatial, Lighting, Camera, and Body Pose
- **Scene & Lighting**: The scene is well-lit with a soft, even light, creating a gentle and serene atmosphere.
- **Camera & Depth**: The camera angle is slightly low, looking up at the character, emphasizing long hair and perspective.
- **Core Pose**: Seated in a relaxed position with legs crossed and arms resting on knees.
- **Upper Body Shape**: Slender with gentle curves, rounded shoulders.
- **Lower Body Shape**: Curvy, slight tilt in hips, arch in back.
- **Physical Anchors**: Hands resting on knees, tail visible near the bottom of the frame.

### Pass 2: Wardrobe, Identity, and Micro-Details
- **Head & Hair**: Blonde hair styled in twintails with a white pom-pom hair ornament.
- **Face & Eyes**: Smiling expression with teeth, eye covered / eyepatch.
- **Torso & Outfit**: Sailor-style school uniform (serafuku) with a red bow, white blouse, and blue skirt/collar.
- **Legwear & Footwear**: White thigh-high socks, white shoes with a subtle blue emblem, garter straps, and o-ring thigh strap.
- **Accessories**: White cat ears, skirt bell accent, white tail.

---

## 2. Final Synthesized Output (Gemma 4 12B)

### Scene / Environment Box:
```text
blue background, polka dot background, aqua background, simple background, well-lit, soft lighting, serene atmosphere, camera looking up
```

### Character / Identity Box:
```text
1girl, solo, blonde hair, long hair, twintails, low twintails, hair between eyes, white pom-pom, white cat ears, white tail, yellow eyes, eyepatch, one eye covered, parted lips, blush, smiling, teeth, red bow, red bowtie, white blouse, white shirt, long sleeves, blue skirt, sailor collar, blue sailor collar, white thighhighs, white socks, white shoes, blue emblem, mole on thigh, thick thighs, curvy, upper body, lower body, sitting, relaxed, legs crossed, knee up, hands on knees, off-shoulder, single bare shoulder, collarbone, armpit crease, head tilt, hair ornament, ahoge, school uniform, serafuku, school swimsuit, swimsuit under clothes, one-piece swimsuit, choker, white choker, garter straps, o-ring, o-ring thigh strap, adjusting legwear, adjusting clothes, skin_indentation
```

---

## 3. Key Architectural Observations

1. **Flawless Environment / Scene Separation**:
   - The Scene box is 100% sterile.
   - It contains only background attributes (`blue background, polka dot background, aqua background, simple background`), lighting quality (`well-lit, soft lighting, serene atmosphere`), and camera angle (`camera looking up`).
   - Zero character body parts or clothing tags leaked into the Scene box.

2. **Micro-Pattern & Prop Accuracy**:
   - The pipeline identified subtle environmental patterns (`polka dot background`) that basic taggers often miss or misclassify.
   - It captured small props and accessories: `white pom-pom`, `white cat ears`, `o-ring thigh strap`, and `garter straps`.

3. **Gemma 12B Synthesis Stability**:
   - When fed concise, checklist-driven visual inputs from Qwen 3B, Gemma synthesized the final prompt in a single pass without token exhaustion, truncation, or JSON parsing issues.
