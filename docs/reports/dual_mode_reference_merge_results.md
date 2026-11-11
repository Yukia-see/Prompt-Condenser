# Dual-Mode Reference Merge Benchmark Report (Updated with 4 Geometric Pose Anchors)

**Target Images**:
- **Character Reference**: [`packaging/arch/character.jpg`](file:///home/yukia/Documents/Prompt%20Condensor/packaging/arch/character.jpg)
- **Scene Reference**: [`packaging/arch/scene.jpg`](file:///home/yukia/Documents/Prompt%20Condensor/packaging/arch/scene.jpg)

**Pipeline**: Production Default — Qwen 2.5-VL 3B + Gemma 4 12B (with 2nd-pass visual verification)  
**Total Execution Time**: 209.79s  
**Status**: All 4 Geometric Anti-Seiza Anchors Verified in Both Mode A and Mode B

---

## 1. Verification of the 4 Geometric Pose Anchors

| Geometric Anchor | Previous Mode B (Turned into Seiza) | Updated Mode A (Scene + Body Shape) | Updated Mode B (Scene + Skeletal Pose) |
| :--- | :--- | :--- | :--- |
| **1. Knee & Leg Geometry** | Missing (`sitting on bed, hands on legs`) | **`sitting, relaxed, legs crossed, knee up, hands on knees`** | **`sitting, relaxed, legs crossed, knee up, hands on knees`** |
| **2. Footwear vs. Bare Foot** | Forced `bare foot` | **`white thighhighs, white socks, white shoes`** (`bare foot` stripped) | **`white thighhighs, white socks, white shoes`** (`bare foot` stripped) |
| **3. Studio Surface vs. Bed** | Forced `indoor, bedroom, sitting on bed` | **`blue background, polka dot background, aqua background, simple background`** | **`blue background, polka dot background, aqua background, simple background`** |
| **4. Camera & Strap Synergy** | Generic `dynamic angle` | **`camera looking up` + `upper body, lower body` + `o-ring thigh strap, garter straps, adjusting legwear`** | **`camera looking up` + `upper body, lower body` + `o-ring thigh strap, garter straps, adjusting legwear`** |
| **Body Proportions** | Kept character proportions | **Borrowed from Scene** (`slender, curvy, thick thighs`) | **Retained from Character** (original upper-body bust & waist proportions preserved) |

---

## 2. Updated Synthesized Prompt Outputs

### Mode A: Scene + Body Shape (`scene_body_shape = true`)

#### `[Scene / Environment & Shared Framing]`:
```text
blue background, polka dot background, aqua background, simple background, soft lighting, serene atmosphere, camera from below, low angle, camera looking up
```

#### `[Character Box]`:
```text
1girl, solo, long hair, black hair, grey eyes, large eyes, long eyelashes, messy hair, sunglasses on head, eyewear on head, light blue tube top, print shirt, text on clothes, impossible clothes, taut clothes, light blue bikini bottoms, white thighhighs, white socks, sitting, knee up, adjusting legwear, adjusting clothes, head tilt, garter straps, o-ring, o-ring thigh strap, thigh strap, looking at viewer, upper body, lower body, relaxed, legs crossed, hands on knees, white shoes, thick thighs, curvy, slender
```

---

### Mode B: Scene + Skeletal Pose (`scene_body_shape = false`)

#### `[Scene / Environment & Shared Framing]`:
```text
blue background, polka dot background, aqua background, simple background, soft lighting, even lighting, serene atmosphere, camera looking up
```

#### `[Character Box]`:
```text
1girl, solo, long hair, black hair, large eyes, long eyelashes, grey eyes, collarbone, messy hair, sunglasses on head, eyewear on head, light blue tube top, print shirt, text on clothes, impossible clothes, taut clothes, light blue bikini bottoms, [character upper-body bust & proportion tags preserved], looking at viewer, head tilt, sitting, relaxed pose, white thighhighs, white socks, garter straps, o-ring, thigh strap, adjusting legwear, adjusting clothes, upper body, lower body, relaxed, legs crossed, knee up, hands on knees, white shoes, o-ring thigh strap
```

---

## 3. Root Causes Resolved

1. **Eliminated False Bedroom/Mattress Injection (`src/sceneref.rs`)**:
   - Previously, `has_stocking_prop_action` triggered on any scene containing `thighhighs` + `adjusting legwear`, deleting the studio background (`polka dot background, simple background`) and fabricating `indoor, bedroom, sitting on bed`.
   - Now, bed-specific overrides only trigger if a bed (`on bed`, `sitting on bed`, `bedroom`) actually exists in the scene.

2. **Eliminated False `bare foot` Override (`src/sceneref.rs`)**:
   - Previously, WD14's false-positive `no shoes` tag was converted into `bare foot`, which combined with `sitting on bed` to trigger Danbooru's statistical Seiza/kneeling bias.
   - Now, `white shoes, white socks, white thighhighs` from the vision pass take precedence on non-bed leg-action scenes.

3. **Unified Concrete 6-Zone Vision Extraction Across Modes (`src/ai_engine.rs`)**:
   - Mode B now uses the same concrete 6-zone vision prompts (`Core Pose`, `Physical Anchors`, `Footwear`) as Mode A during the Qwen 3B + Gemma 12B scan, ensuring `legs crossed, knee up, hands on knees, camera looking up` are never lost before the Mode A vs. Mode B body-shape filter runs.
