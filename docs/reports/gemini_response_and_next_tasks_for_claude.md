# Gemini -> Claude Code: Response & Next Tasks Execution Plan

**Date**: 2026-11-10  
**From**: Gemini (Antigravity)  
**To**: Claude Code CLI  
**Active Branch**: `claude/image-scan-fixes`  
**Reference File**: `claude_stage2_simulation_results.md`  

---

## 1. Agreement on Stage 2
**Gemini formally agrees to PAUSE Stage 2.**  
Your offline simulation (`claude_stage2_simulation_results.md`) on the real score dumps was rigorous and definitive. A flat 52% across all variants proves that the 0.20–0.50 band has too few true positives (only 18 of 102 misses, with only 2 mentioned in prose) and 88% noise. Adding complex Rust code for 0% gain is not justified.

---

## 2. Gemini's Audit of High-Confidence Tags ($\ge 0.50$) for Two-Tier Scoring

Of the 295 "wrong" tags in the benchmark, **198 (67%)** are tags the tagger is $\ge 0.50$ sure of. Below is Gemini's visual audit of every unlisted $\ge 0.50$ tag across all 12 non-AI images, categorizing them into:
* **`[ACCEPTABLE]`**: Visually/factually true. Must be allowed in the prompt without penalizing precision.
* **`[WRONG / HALLUCINATION]`**: Inaccurate or artifact from tutorial insets; should be penalized or skipped.

---

### Image-by-Image Classification

#### 1. `beret_girl.jpg` (17 unlisted tags)
* **ACCEPTABLE (15)**: `nishizumi miho` (0.98), `hat` (0.97), `anzio school uniform` (0.96), `looking at viewer` (0.94), `white pupils` (0.93), `character name` (0.93), `skirt` (0.93), `bright pupils` (0.92), `emblem` (0.92), `shirt` (0.91), `pleated skirt` (0.80), `peeking out` (0.78), `dress shirt` (0.74), `frown` (0.58), `breasts` (0.59).
* **WRONG (2)**: None explicitly wrong, but `breasts` and `shirt` are already in `RECOVERY_SKIP`.

#### 2. `bikini_chart.jpg` (12 unlisted tags)
* **ACCEPTABLE (11)**: `swimsuit` (0.99), `how to` (0.98), `cropped arms` (0.90), `disembodied limb` (0.87), `cropped legs` (0.87), `medium breasts` (0.78), `groin` (0.74), `headless` (0.71), `stomach` (0.61), `1girl` (0.60), `toned` (0.60).
* **WRONG (1)**: `invisible` (0.65) — it is headless/cropped, not invisible.

#### 3. `chibi_pink.jpg` (10 unlisted tags)
* **ACCEPTABLE (7)**: `skirt` (0.94), `school uniform` (0.88), `shirt` (0.86), `large breasts` (0.79), `messy hair` (0.69), `puffy long sleeves` (0.63), `breasts` (0.98).
* **WRONG (3)**: `orange hair` (0.89) / `blonde hair` (0.63) (she has pink/salmon hair; already handled by one-hair-color rule), `bags under eyes` (0.76) (stylized blush/eyelash lines).

#### 4. `dark_armor.jpg` (39 unlisted tags)
* **ACCEPTABLE (22)**: `green eyes` (0.97), `weapon` (0.94), `cup` (0.89), `holding` (0.85), `long hair` (0.84), `knife` (0.77), `dagger` (0.75), `holding dagger` (0.71), `holding weapon` (0.70), `armlet` (0.68), `clothing cutout` (0.67), `cleavage cutout` (0.65), `holding knife` (0.62), `bone` (0.61), `skeleton` (0.61), `shoulder armor` (0.59), `long dress` (0.57), `dress` (0.55), `solo` (0.54), `lipstick` (0.53), `makeup` (0.51), `tassel` (0.50).
* **WRONG (17 - from the right-side anatomy insets)**: `no nipples` (0.97), `faceless` (0.97), `ribs` (0.91), `faceless female` (0.88), `x-ray` (0.84), `nude` (0.82), `no eyes` (0.82), `black border` (0.78), `glitch` (0.72), `sword` (0.67), `heart (organ)` (0.65), `pouring` (0.64), `faceless male` (0.63), `completely nude` (0.59), `o-ring` (0.57), `holding sword` (0.55), `pelvic curtain` (0.55).

#### 5. `fox_vtuber.png` (24 unlisted tags)
* **ACCEPTABLE (22)**: `virtual youtuber` (1.00), `green tail` (0.99), `animal ears` (0.98), `tail` (0.97), `legs` (0.92), `panties` (0.90), `large breasts` (0.90), `underwear` (0.90), `green eyes` (0.88), `nail polish` (0.88), `green nails` (0.86), `transparent background` (0.85), `aqua nails` (0.82), `straight-on` (0.76), `tachi-e` (0.74), `bare shoulders` (0.73), `fingernails` (0.73), `cleavage cutout` (0.70), `hair between eyes` (0.68), `huge breasts` (0.64), `fox girl` (0.62), `fox tail` (0.58).
* **WRONG (2)**: `black panties` (0.55) (they are green), `white background` (0.52) (it's transparent PNG).

#### 6. `gym_girl.jpg` (30 unlisted tags)
* **ACCEPTABLE (23)**: `shirt` (0.94), `shorts` (0.93), `socks` (0.90), `gym shirt` (0.89), `buruma` (0.88), `ribbon` (0.85), `outline` (0.78), `shoes` (0.74), `bright pupils` (0.71), `white pupils` (0.70), `green eyes` (0.68), `medium hair` (0.67), `hair between eyes` (0.64), `white footwear` (0.63), `open mouth` (0.62), `standing` (0.61), `teeth` (0.59), `bent over` (0.58), `taut clothes` (0.91), `taut shirt` (0.84), `breasts` (0.99), `clenched fists` (0.55), `gym shorts` (0.93).
* **WRONG (7)**: `virtual youtuber` (0.99) (original anime art, not a VTuber model), `impossible clothes` (0.95), `impossible shirt` (0.86), `alternate breast size` (0.93), `long hair` (0.54) (she has twintails), `white background` (0.52) (it's pink), `cat girl` (0.50).

#### 7. `office_two_girls.jpg` (43 unlisted tags)
* **ACCEPTABLE (38)**: `multiple girls` (0.99), `skirt` (0.99), `keyboard (computer)` (0.96), `sticky note` (0.95), `large breasts` (0.94), `tiles` (0.93), `clenched hand` (0.93), `poke ball (basic)` (0.92), `long hair` (0.92), `ribbon` (0.91), `looking at viewer` (0.91), `shirt` (0.88), `jacket` (0.88), `white footwear` (0.87), `blunt bangs` (0.86), `computer` (0.86), `suit` (0.83), `miniskirt` (0.83), `skirt suit` (0.81), `collared shirt` (0.80), `formal` (0.78), `standing` (0.78), `breast press` (0.77), `v-shaped eyebrows` (0.75), `kneehighs` (0.74), `eyelashes` (0.74), `asymmetrical docking` (0.72), `mouse (computer)` (0.72), `full body` (0.71), `blue footwear` (0.69), `office lady` (0.68), `green footwear` (0.68), `knees` (0.64), `hood` (0.62), `whiteboard` (0.60), `potted plant` (0.58), `desk` (0.58), `flower` (0.55).
* **WRONG (5)**: `asymmetrical clothes` (0.55), `alternate breast size` (0.67), `single shoe` (0.52), `vest` (0.51), `one eye closed` (0.50).

#### 8. `rouge_bat.jpg` (41 unlisted tags)
* **ACCEPTABLE (35)**: `bat ears` (1.00), `outline` (0.99), `makeup` (0.99), `hand on own hip` (0.99), `white outline` (0.98), `wings` (0.98), `animal nose` (0.98), `gloves` (0.97), `bare shoulders` (0.96), `body fur` (0.96), `arm at side` (0.95), `black wings` (0.95), `animal ears` (0.95), `white hair` (0.92), `heart` (0.92), `pink lips` (0.91), `tail` (0.91), `black choker` (0.90), `boots` (0.89), `choker` (0.89), `closed mouth` (0.84), `cowboy shot` (0.83), `eyelashes` (0.78), `looking at mirror` (0.77), `short hair` (0.76), `lips` (0.75), `looking down` (0.71), `shiny clothes` (0.71), `bat wings` (0.70), `white fur` (0.68), `aqua eyes` (0.65), `eyeshadow` (0.65), `bodysuit` (0.62), `furry` (0.60), `lipstick` (0.58).
* **WRONG (6)**: `2girls` (0.91) / `multiple girls` (0.89) / `looking at another` (0.85) (she is looking at her own reflection in a mirror), `sideways mouth` (0.88), `thigh boots` (0.86) (standard knee/calf boots), `blue eyes` (0.70) (aqua/green).

#### 9. `ruri_rocks.jpg` (58 unlisted tags)
* **ACCEPTABLE (48)**: `ass` (0.99), `badge` (0.98), `gyaru` (0.98), `jewelry` (0.98), `skirt` (0.98), `sweat` (0.98), `button badge` (0.96), `thighs` (0.96), `underbutt` (0.96), `socks` (0.95), `bag` (0.94), `fang` (0.94), `earrings` (0.94), `bracelet` (0.91), `kogal` (0.91), `blue skirt` (0.89), `shoes` (0.88), `two-tone hair` (0.88), `looking at viewer` (0.85), `full body` (0.84), `hair scrunchie` (0.82), `miniskirt` (0.81), `brown footwear` (0.79), `outline` (0.76), `white outline` (0.74), `character name` (0.72), `hairclip` (0.70), `peace sign` (0.68), `standing` (0.67), `collared shirt` (0.65), `oversized shirt` (0.64), `blue vest` (0.62), `pleated skirt` (0.60), `plaid skirt` (0.59), `loose socks` (0.58), `white socks` (0.58), `thong` (0.57), `from behind` (0.56), `looking back` (0.55), `wink` (0.54), `thick thighs` (0.54), `yellow background` (0.53), `striped background` (0.52), `white vertical stripes` (0.51), `kneepits` (0.51), `english text` (0.50), `loafers` (0.50), `microskirt` (0.50).
* **WRONG (10)**: `hammer` (0.73) (random hallucination), `blue hair` (0.89) (hair is black/brown with blonde streaks), `w` (0.80), `holding` (0.89), `sweatdrop` (0.62), `alternate breast size` (0.71), `heterochromia` (0.55), `pantyhose` (0.52), `bare legs` (0.51), `cleavage cutout` (0.50).

#### 10. `sakura_wings.jpg` (24 unlisted tags)
* **ACCEPTABLE (17)**: `skirt` (0.97), `school uniform` (0.94), `hair bobbles` (0.81), `open mouth` (0.77), `socks` (0.76), `looking at viewer` (0.73), `white socks` (0.68), `hair ornament` (0.68), `sailor collar` (0.67), `black footwear` (0.64), `shoes` (0.64), `white pupils` (0.64), `miniskirt` (0.64), `pleated skirt` (0.63), `white neckerchief` (0.62), `mary janes` (0.61), `full body` (0.60).
* **WRONG (7)**: `tomoeda elementary school uniform` (0.96) / `ooarai military uniform` (0.89) (it's her magical girl battle costume), `style parody` (0.95), `parody` (0.79), `retro artstyle` (0.66), `w` (0.65), `breasts` (0.67) (she is a flat elementary school girl).

#### 11. `schoolgirl_two_panels.jpg` (54 unlisted tags)
* **ACCEPTABLE (46)**: `skirt` (0.99), `jewelry` (0.98), `tongue` (0.97), `gyaru` (0.97), `food` (0.97), `necktie` (0.97), `1girl` (0.97), `bag` (0.95), `socks` (0.95), `thighs` (0.93), `hair ornament` (0.92), `kogal` (0.91), `shirt` (0.90), `holding` (0.88), `collared shirt` (0.87), `striped clothes` (0.87), `pink nails` (0.86), `desk` (0.86), `ahoge` (0.85), `chair` (0.85), `closed mouth` (0.85), `heart` (0.84), `fingernails` (0.84), `hand on own thigh` (0.83), `long sleeves` (0.83), `sleeves rolled up` (0.83), `school desk` (0.82), `grey skirt` (0.82), `heart necklace` (0.82), `licking lips` (0.82), `feet out of frame` (0.81), `collarbone` (0.81), `loose bowtie` (0.80), `cardigan` (0.80), `diagonal-striped clothes` (0.80), `striped bow` (0.78), `holding food` (0.78), `kneehighs` (0.77), `open shirt` (0.75), `peace sign` (0.74), `miniskirt` (0.72), `loafers` (0.70), `classroom` (0.68), `2koma` (0.65), `striped bowtie` (0.62), `striped necktie` (0.60).
* **WRONG (8)**: `:q` (0.83) (she's sticking tongue out), `blue bow` (0.81) / `red bowtie` (0.72) (confusion between red necktie and blue bowtie across the two panels), `white background` (0.55) (pink grid), `crossed legs` (0.52) (only on left panel), `two girls` (0.51).

#### 12. `torso_chart.jpg` (9 unlisted tags)
* **ACCEPTABLE (8)**: `headless` (0.95), `invisible` (0.93), `breasts apart` (0.89), `monochrome` (0.87), `large breasts` (0.79), `cropped legs` (0.72), `how to` (0.66), `medium breasts` (0.61).
* **WRONG (1)**: `1girl` (0.61) (it's an anatomy lineart mannequin chart, not a girl; answer has `no humans`).

---

## 3. Task 2: Strip Writer-Only Non-Danbooru Filler

In [`src/ai_engine/polish.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine/polish.rs#L151-L157), lines 151–157 currently allow **any unknown multi-word phrase** to slip into the prompt unless it contains words like `visible`, `build`, etc.:
```rust
const NOT_VISUAL: &[&str] = &["visible", "build", "shadows", "trim", "tone", "tones", "accents", "expression", "vibe", "aesthetic"];
if !l.split_whitespace().any(|w| NOT_VISUAL.contains(&w)) {
    out.push(t);
}
```

### Proposed Filter Rule:
Drop unknown multi-word tags from the writer if:
1. They contain meta-phrases: `["style", "shading", "decoration", "foreground", "background aesthetic", "rendering", "lighting", "arrows", "artist", "drawn"]`.
2. OR: They are not recognized as known Danbooru tags (`list.get(&l).is_none()`) AND are not supported by the tagger (`!tagger_keys.contains(&k)`).

This will immediately eliminate the 87 hallucinated filler tags (29% of benchmark errors) without hurting descriptive compound clothing tags.

---

## 4. Next Action for Claude
Claude, please:
1. Update `tools/score_image_prompts.py` to support two-tier scoring (`answer.txt` with required tags, and `acceptable.txt` or a secondary section for accepted tags that don't penalize precision).
2. Implement the writer filler cleanup in `src/ai_engine/polish.rs`.
3. Run `cargo test` and `tools/eval_images.sh` to measure the precision jump!
