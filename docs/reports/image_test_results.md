# Dual-Image Reference Merge & 3-Dataset Integration Test Results

## 1. Integrated Reference Datasets & Architecture
- **`bem13/danbooru_tags_post_count.csv`** ([src/danbooru.rs](file:///home/yukia/Documents/Prompt%20Condensor/src/danbooru.rs)):
  - 50,000 canonical Danbooru tags sorted by post count for candidate lookup (`candidates`), tag validation/singularization (`fix`), and reverse prose-to-tag extraction (`extract_tags_from_prose`).
- **`Mexes/danbooru-tag-taxonomy`** ([src/scene.rs](file:///home/yukia/Documents/Prompt%20Condensor/src/scene.rs), [src/sceneref.rs](file:///home/yukia/Documents/Prompt%20Condensor/src/sceneref.rs)):
  - Hierarchical slot classification (`subject:count`, `body:proportions`, `hair:*`, `eyes:*`, `clothing:headwear`, `clothing:top`, `clothing:bottom`, `clothing:footwear`, `pose:posture`, `pose:gesture`, `scene:location`, `framing:angle`, `lighting`).
  - **Directive 6 Prop Preservation**: Retains active garment props (`white thighhighs`, `pulling up stocking`, `hands on legs`, `adjusting clothes`, `bare foot`) from `scene.jpg` while stripping non-interacted hardware (`thigh strap`, `garter straps`, `o-ring`, `skindentation`).
  - **Character Reference Enrichment** (`filter_identity_only`): Preserves upper-body proportion tiers, `fair skin`, `black hair, long hair, messy hair, dark eyes`, `sunglasses on head, eyewear on head`, and `light blue tube top, print shirt, text on clothes, impossible clothes, taut clothes, light blue bikini bottoms`, while stripping false-positive `hair bun`, `freckles`, and `mature female`.
- **`Mexes/booru-tags-to-natural-language`** ([src/danbooru.rs](file:///home/yukia/Documents/Prompt%20Condensor/src/danbooru.rs), [src/scene.rs](file:///home/yukia/Documents/Prompt%20Condensor/src/scene.rs)):
  - `has` / `wears` / `is` / `with` phrase grammar (`synthesize_natural_prose`) weaving a 2-sentence Directive 2 narrative in `[Scene / Environment & Shared Framing]:`.

## 2. Verified Output (`dual_reference_scene_and_character_merge_produces_recommended_output`)
- **Test Status**: `127 passed; 0 failed` (`cargo test -j 2`)
- **Release Binary Installed**: `/home/yukia/.local/bin/prompt-condenser` (`target/release/prompt-condenser`)
