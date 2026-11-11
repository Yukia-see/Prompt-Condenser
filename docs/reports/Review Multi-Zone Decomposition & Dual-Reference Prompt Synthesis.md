# Review: Multi-Zone Decomposition & Dual-Reference Prompt Synthesis

Nov 8, 2026 · @Yuka

## Verdict

The core idea is sound: cropping so WD14 can see small items (hairpins, held objects, footwear), and keeping the scene and character prompts separate, beats a single full-frame pass. But the document claims more than it shows, a few of its technical statements are wrong, and some of its mechanisms would add wrong tags.

## Conflict with the pipeline we built

The document describes a different pipeline from the Python and Rust code, so decide which one is the real one before handing anything to Antigravity.

|  | This document | Python and Rust code |
| --- | --- | --- |
| Vision model | Qwen 2.5-VL 3B | Moondream |
| Input to the vision model | Crops: 4 quadrants plus 3 zones per figure | Whole image, two prompts |
| Detector | YOLO | None |
| Vision temperature | 0.05 | 0.1 |
| Synthesis temperature | 0.10 | 0.35 |
| Meaning of "dual" | Two input images (scene plus character) | Two passes over the same image |

No crop stage exists in the code I wrote, and it does no two-image merge.

## Problems that would produce wrong output

Six mechanisms in the document would add or delete tags incorrectly.

1. **Crops create their own tags.** WD14 on a cropped region emits tags such as `head out of frame`, `cropped legs`, `portrait`, `close-up` and `upper body`. Those describe the crop, not the image. Framing, camera, lighting and global-background tags should come from the full frame only. Rating and character-name tags should never come from crops. The `max(base, crop)` rule would let all of these through.
2. **The tube top example is wrong.** The conflict matrix lists `tube top` as a closed neckline that cancels `cleavage`. Tube tops are strapless and commonly show cleavage, so the rule would delete correct tags. Boots versus socks has the same flaw, because socks inside boots are legitimate.
3. **The three zones assume a full body.** A portrait, or a figure cut off by the image edge, has no real lower zone, and cropping there invites hallucination. The zones also overlap (0–55%, 30–70%, 55–100%), so one item can be counted twice and look more confident than it is.
4. **Overlapping characters still bleed.** With two girls hugging, each bounding box contains part of the other. The claim that one character's hairpin is never attributed to the other is false without masking or segmentation.
5. **Splitting the prompt into two strings does not stop bleed.** Cross-attention bleed remains unless the generator uses regional conditioning (Proposal 6). Concatenated, the two strings are the same prompt.
6. **The aspect-ratio auto-negative is risky.** Injecting `kneeling, seiza` as negatives would break images that really show kneeling, so it should be gated on the detected pose. Negatives also do nothing for models run without CFG, which includes Flux, the target the document promotes elsewhere.

## Statements that are wrong or unsupported

Several claims have no evidence behind them, and a few are technically incorrect.

- **"Zero hallucination" at temperature 0.05:** temperature does not prevent hallucination, and small crops make a 3B model invent things. `repeat_penalty 1.15` can also distort tag-style output.
- **"29 new tags":** this comes from one image, with no check that the tags are correct. "Unrivaled" and "state-of-the-art" have no support either.
- **"300% increased risk":** this number has no source.
- **"Physically prevent" kneeling drift:** text tags only nudge the generator. For firm pose control, exporting a skeleton for ControlNet or DWPose works better than any wording.
- **Qwen "downsampled to 1024×1024":** as far as I know, Qwen2.5-VL uses dynamic resolution, so the amplification argument is much weaker for it than for WD14, which takes a fixed 448×448 input.
- **"4 anchors":** five items are listed, and one of them, `white shoes`, is not geometric. The "8-bucket hierarchy" and the "triggers" are never defined.

## Problems inside the proposals

Proposals 1, 2 and 5 each rest on a mistaken assumption.

| Proposal | Problem |
| --- | --- |
| 1: YOLOv8-Pose keypoints | The 17 COCO keypoints include ankles but no toes, so they cannot give toe or footwear boxes. A wrist point is not a hand box. A 2D hip-knee-ankle angle cannot prove a knee is raised, because of foreshortening. COCO-trained models also do poorly on stylized art, so check whether the detector is anime-capable. DWPose is probably a better fit and gives hands as well. |
| 2: Active-query boxes | As far as I know, Qwen2.5-VL returns absolute pixel coordinates, not the normalized `[ymin, xmin, ymax, xmax]` the document asks for. Every box it proposes should be confirmed by an independent model (WD14) before its tags are accepted. |
| 5: Attention-weighted scoring | It needs a list of which tag categories are eligible from crops. Without that list, the weights only soften the crop-created-tags problem instead of fixing it. |

## How to improve it, in priority order

Start by measuring, then fix the tag sources, then add the larger features.

1. **Build a small evaluation first.** Use 30 to 50 images with verified tags and measure precision and recall, with an ablation per component. Also compare how often seiza drift happens with and without the anchors. Right now "better" cannot be measured.
2. **Add a per-crop tag policy.** Keep a whitelist of micro-attribute tags allowed from crops and a blacklist of framing, crop-artifact, rating and character tags. The full frame is the only source for global tags.
3. **Require two sources.** A VLM-only tag survives only if WD14 confirms it, or if a second look at a different crop agrees.
4. **Handle detection edge cases.** Skip zones the figure does not have, and mask the neighbor when boxes overlap.
5. **Replace the conflict matrix with a layer model.** Each tag gets a body region and a layer (outer, inner, footwear, legwear). A layer that covers another hides it instead of deleting it.
6. **Do pose export together with Proposal 1.** The same keypoints feed both the crops and the ControlNet skeleton.
7. **Set a performance budget.** A figure costs about 1 + 4 + 3N WD14 calls plus the VLM calls. Qwen and Gemma at 128K context probably will not both stay in VRAM, so Ollama would reload models every run. Batch the WD14 calls and lower the context window.

## Suggested next step

Add a crop stage to the Rust module and the Python reference, with the per-crop tag policy and tests, kept in step by the existing parity setup. Start with the tag policy, because it fixes wrong output the crops would produce today.
