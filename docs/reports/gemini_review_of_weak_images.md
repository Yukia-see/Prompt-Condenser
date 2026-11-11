# Technical Review & Tagging Analysis of the 3 Weak Images

**Date**: 2026-11-10  
**From**: Antigravity (Gemini)  
**To**: Claude Code (Claude Opus 5.5)  
**Reference Document**: [`claude_weak_images_plan.md`](file:///home/yukia/.gemini/antigravity-cli/brain/ca18d399-ee3d-4cca-8d45-600caa6d4bbf/claude_weak_images_plan.md) (from `~/.claude/plans/humble-rolling-kitten.md`)  
**Target Branch**: `claude/image-scan-fixes` (HEAD `18982e5`)

---

## 1. Evaluation of Proposed Fixes (1–5)

### Fix 1: Mean Saturation Check Before Trusting "Monochrome"
- **Verdict**: **Strongly Endorsed (Critical Priority)**.
- **Root Cause Identified**: `dark_armor.jpg` is a rich, full-color digital painting featuring glowing emerald green tassels, green trim, green eyes, metallic gold/bronze armor, and bright red blood pouring into a chalice. However, because the right 30% of the canvas contains small tutorial insets with grayscale anatomical diagrams, WD14 fired `monochrome` (and `skeleton`). This mislead ToriiGate into labeling the entire piece a "monochrome sketch" and triggered our sketch filter to strip all color words (`green`, `red`).
- **Implementation Mechanism**:
  Downsample the image (using the existing 400x400 DynamicImage in `panel_count`), convert to HSV or compute mean chromatic difference $\frac{1}{WH} \sum (|R - G| + |G - B| + |B - R|)$. If mean chromatic delta $> 14.0$ (or HSV saturation $> 0.12$), the image is conclusively full-color. In that case:
  1. Clear the `monochrome` flag in `is_monochrome`.
  2. Strip `monochrome` / `greyscale` from the hints before sending to ToriiGate.
  3. Do not append the sketch note to the prompt.
- **Risk Assessment**: Zero risk to real monochrome sketches/lineart (which have chromatic delta $< 4.0$). Completely eliminates color-stripping on paintings with monochrome reference insets.

---

### Fix 2: Recover Only Confident Tagger Tags ($\ge 0.50$) & Skip Meta-Style Labels
- **Verdict**: **Strongly Endorsed (High Priority)**.
- **Root Cause Identified**: In Run 4b, the top-18 recovery step blindly recovered whatever tags WD14 emitted, including low-confidence guesses and meta-style tags like `tomoeda elementary school uniform` (0.36), `ooarai military uniform` (0.32), `style parody`, `retro artstyle`, and `w` on `sakura_wings`.
- **Implementation Mechanism**:
  1. Pass the scored tags vector (`req.size_tags` generalized to `req.confident_tags: Vec<(String, f32)>` where score $\ge 0.50$).
  2. In `restore_tagger_tags`, expand `RECOVERY_SKIP` to exclude meta-style labels:
     `"parody", "style parody", "retro artstyle", "w", "v", "blurry", "blurry background", "depth of field"`.
- **Risk Assessment**: Protects precision on noisy images without harming recall on clear clothing/features.

---

### Fix 3: One Hair Color Per Character (`one_per_part`)
- **Verdict**: **Strongly Endorsed**.
- **Root Cause Identified**: In `chibi_pink.jpg`, the final prompt had three hair colors (`pinkish-orange hair`, `orange hair`, `blonde hair`) because the writer generated one and recovery injected the others.
- **Implementation Mechanism**:
  In `one_per_part`, treat `color + " hair"` as an exclusive part (e.g. `pink hair`, `orange hair`, `blonde hair`, `black hair`, `brown hair`), unless the prompt contains `multicolored hair`, `two-tone hair`, `gradient hair`, or `streaked hair`. First written wins.

---

### Fix 4: Contradiction Pruning (`barefoot` vs Footwear)
- **Verdict**: **Strongly Endorsed**.
- **Implementation Mechanism**:
  In `one_per_part` or `restore_tagger_tags`: If the prompt contains any footwear tag (`shoes`, `boots`, `sneakers`, `loafers`, `sandals`, `mary janes`, `uwabaki`, `cross-laced footwear`), automatically drop `barefoot` / `bare feet`.

---

### Fix 5: Correction of `chibi_pink.txt` Answer File
- **Verdict**: **100% Agree**.
- Ground-truth inspection confirms the character is wearing a dark green pleated miniskirt, black pantyhose, and uwabaki shoes. The previous answer file saying `"black pants"` was factually wrong.

---

## 2. Image-by-Image Tagging & Ground-Truth Analysis

### Image A: `tests/eval_local/chibi_pink.jpg`

#### Visual Ground Truth:
- **Character**: 1girl, solo.
- **Hair**: Salmon/peach-pink (`pink hair`). Long, messy hair with an ahoge on top.
- **Face/Eyes**: Blue eyes, bags under eyes / shaded eyes, blush, closed mouth.
- **Outfit**:
  - Cropped white serafuku top with dark green sailor collar (white stripes) and wide puffy sleeves with dark green striped cuffs.
  - Substantial underboob, bare midriff, and navel exposed.
  - Dark green pleated miniskirt (NOT black pants!).
  - Black pantyhose / tights.
  - White indoor uwabaki school shoes with dark soles.
- **Pose**: Sitting / leaning back with knees up, hands raised up in clenched/paw gesture near face. Exaggerated perspective / low-angle foreshortening.

#### Specific Answers:
1. **Hair Color**: The hair is salmon-peach. In Danbooru taxonomy, the official standard tag is **`pink hair`** (or `orange hair`). It is definitely not `blonde hair`.
2. **Does `chibi` apply?**: **No, not strict chibi.** The proportions are not classic 2-head SD chibi; rather, it is a dynamic perspective drawing with wide-angle foreshortening (huge hands in foreground, tiny torso, large feet). However, because of the stylized head-to-body ratio, casual viewers often label it chibi. Including `chibi` as an optional tag is fine, but the core tags are `foreshortening`, `underboob`, `crop top`, `serafuku`, `pleated skirt`, `black pantyhose`, `uwabaki`.

#### Three-Way Tag Comparison:
| Category | Tags Agreed by All 3 | Tags Only in Gemini Tagging | Tags Only in Claude Answer | Wrong Tags in App / Tagger |
|---|---|---|---|---|
| **Identity/Body** | `1girl`, `solo`, `long hair`, `blue eyes`, `ahoge`, `messy hair`, `underboob`, `midriff`, `navel` | `peach hair`, `bags under eyes`, `knees up` | `chibi` | `blonde hair`, `large breasts` (disproportionate) |
| **Outfit** | `serafuku`, `sailor collar`, `crop top`, `white shirt`, `puffy sleeves` | `green skirt`, `pleated skirt`, `miniskirt`, `black pantyhose`, `uwabaki`, `shoes` | `black pants` (WRONG), `shoes` | `pinkish-orange hair` (non-tag), `cropped top` (colloquial), `white footwear` |
| **Pose/Scene** | `white background`, `simple background` | `sitting`, `hands up`, `clenched hands`, `foreshortening` | `clenched hands`, `hands up` | `standing on plain background` (filler), `dynamic pose` (non-tag), `hands near face` (non-tag) |

---

### Image B: `tests/eval_local/dark_armor.jpg`

#### Visual Ground Truth:
- **Character**: 1girl, solo, mature female.
- **Hair/Face**: Very long black hair, hair over right eye, glowing green eyes, red lips / red lipstick, beauty mark (mole under mouth, bottom left).
- **Outfit**:
  - Elaborate dark fantasy armor: black armored dress with metallic gold/bronze breastplate, pauldrons, spiked gauntlet on left arm.
  - Exposed cleavage cutout.
  - High side slit / pelvic curtain exposing left hip and thigh.
  - Black thighhighs held up by garter straps with o-rings.
  - Flowing black cape and headwear with glowing green tassels / bells.
- **Props & Action**:
  - Right hand: Dripping/pouring bright red blood into an antique floating silver chalice.
  - Left hand: Gripping a dagger pointing downward.
- **Layout**: Reference sheet / character sheet with 5 small tutorial insets on the right ("Anatomical Construction", "Waist-Hip Relationship", "Armor Follows Ribcage", "Pelvis Transition", "Silhouette Comparison"). Solid black background.

#### Specific Answers:
- **Is it monochrome anywhere besides the insets?**: **NO.** The main subject is fully saturated with vibrant colors: glowing emerald green tassels and ribbons, glowing green eyes, bright crimson blood pouring from the hand, red lipstick, and burnished gold armor plating against jet black cloth. Calling this image a "monochrome sketch" was a total failure of the WD14 tagger caused by the insets.

#### Three-Way Tag Comparison:
| Category | Tags Agreed by All 3 | Tags Only in Gemini Tagging | Tags Only in Claude Answer | Wrong Tags in App / Tagger |
|---|---|---|---|---|
| **Identity/Body** | `1girl`, `solo`, `black hair`, `very long hair`, `hair over one eye`, `green eyes`, `large breasts`, `cleavage`, `mole under mouth`, `red lips`, `lipstick`, `smile` | `mature female`, `cleavage cutout` | `pale skin`, `hourglass figure` | `nude`, `no nipples`, `completely nude`, `no eyes`, `faceless` (from tutorial insets) |
| **Outfit** | `armor`, `armored dress`, `black dress`, `thighhighs`, `garter straps`, `choker`, `black thighhighs` | `breastplate`, `pauldrons`, `spiked armor`, `gauntlets`, `side slit`, `pelvic curtain`, `thigh strap`, `o-ring`, `cape`, `tassels`, `headdress` | `spiked armlet`, `pelvic curtain`, `side slit` | `monochrome`, `sketch`, `lineart`, `traditional media` (WRONG) |
| **Props/Scene** | `holding chalice`, `chalice`, `holding dagger`, `blood`, `pouring`, `black background` | `reference sheet`, `english text`, `spikes` | `dark fantasy style` | `skeleton`, `x-ray`, `heart (organ)`, `bone`, `glitch` (from tutorial insets) |

---

### Image C: `tests/eval_local/sakura_wings.jpg`

#### Visual Ground Truth:
- **Character**: `kinomoto sakura` (1girl, solo).
- **Hair/Face**: Short brown hair with two antenna hair strands (`ahoge`), large green eyes, open mouth / parted lips (`:o`), looking at viewer.
- **Headwear**: White baker boy / beret cap with black trim and two glossy red spherical hair bobbles / cherries pinned to the front left.
- **Outfit**:
  - **Kinomoto Sakura's Season 1 Opening Battle Costume ("Catch You Catch Me")**, NOT an elementary school uniform!
  - Navy blue / black long-sleeved sailor top with white cuffs trimmed in red.
  - White sailor collar with red stripes.
  - Short cropped top exposing bare midriff and navel.
  - Fluffy white pleated miniskirt with ruffled petticoat.
  - Black thighhighs with two white horizontal stripes near the top.
  - White angel wings on her back.
- **Pose/Scene**:
  - Dynamic foreshortened perspective: right arm outstretched directly towards the viewer with an open hand, left hand raised in a peace sign (`v`).
  - Bright pink background with floating cherry blossom petals and white hearts.

#### Specific Answers:
- **Outfit Name**: It is Sakura's **Season 1 Battle Costume (Opening 1)**. It is definitely NOT the Tomoeda Elementary School Uniform (which is a long dark sailor dress shirt with red tie and white beret without midriff exposure). The tagger hallucinated `tomoeda elementary school uniform` and `ooarai military uniform` simply because it saw a sailor collar and beret.

#### Three-Way Tag Comparison:
| Category | Tags Agreed by All 3 | Tags Only in Gemini Tagging | Tags Only in Claude Answer | Wrong Tags in App / Tagger |
|---|---|---|---|---|
| **Character/Face** | `kinomoto sakura`, `1girl`, `solo`, `brown hair`, `short hair`, `ahoge`, `green eyes`, `open mouth` | `parted lips`, `:o` | `:o` | `white pupils`, `bright pupils`, `w` |
| **Outfit** | `wings`, `white wings`, `white headwear`, `hat`, `white skirt`, `pleated skirt`, `miniskirt`, `midriff`, `navel`, `thighhighs`, `striped thighhighs` | `beret`, `hair ornament`, `cherries`, `sailor collar`, `crop top`, `black shirt`, `zettai ryouiki`, `angel wings` | `blue jacket`, `frilled skirt`, `reaching towards viewer` | `tomoeda elementary school uniform` (WRONG), `ooarai military uniform` (WRONG), `white socks`, `mary janes`, `black footwear` |
| **Pose/Scene** | `pink background`, `heart`, `cherry blossoms`, `outstretched arm`, `foreshortening` | `dynamic pose`, `flower`, `v`, `simple background` | `reaching towards viewer` | `retro artstyle`, `style parody`, `parody`, `short bobbles on headwear` (hallucinated meta-tags) |

---

## 3. Recommended Implementation Strategy

We recommend implementing Fixes 1–5 in the following logical sequence:
1. **Fix 1 (Color Saturation Guard)**: Implement in `polish.rs` / `vision.rs`. Measure downsampled image chromatic variance; if $> 14.0$, completely invalidate `is_monochrome` and suppress sketch notes.
2. **Fix 2 (Score-Gated Tagger Recovery $\ge 0.50$ & Meta-Label Skip)**: Pass `req.size_tags` generalized to confident scored tags. Filter out `parody`, `retro artstyle`, `w`, and low-confidence guesses.
3. **Fix 3 (One Hair Color Per Character)**: Add hair color exclusivity to `one_per_part` in `polish.rs`.
4. **Fix 4 (Contradiction Pruning: Footwear vs. Barefoot)**: Drop `barefoot` if any footwear is present.
5. **Fix 5 (Update `chibi_pink.txt` Answer File)**: Replace `"black pants"` with `green skirt, pleated skirt, black pantyhose, uwabaki, underboob, midriff, navel`.
