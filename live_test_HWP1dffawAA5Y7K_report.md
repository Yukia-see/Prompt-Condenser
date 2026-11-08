# Live Pipeline Test Report: Phases 0-5 End-to-End Validation

- **Image Tested:** `tests/eval_dataset/HWP1dffawAA5Y7K.jpg` (832x1216)
- **Detection Figure:** `(3.0, 12.0)` to `(832.0, 1215.0)`
- **Figure Height:** `1203.0px`

## 1. Pure Zone Decomposition (Phase 2)

| Zone | Geometry Box (x0, y0, x1, y1) | Target Anatomical Features |
| :--- | :--- | :--- |
| **Upper** | `(0.0, 0.0) -> (832.0, 721.8)` | Upper figure details |
| **Middle** | `(0.0, 324.8) -> (832.0, 902.2)` | Middle figure details |
| **Lower** | `(0.0, 625.5) -> (832.0, 1216.0)` | Lower figure details |

## 2. Wrist-Centered Hand Bounding Boxes (Phase 5)

- **Proportional Hand Radius:** `216.5px` ($18\% \times H_\text{figure}$)
- **Right Hand Box:** `(0.0, 421.0) -> (426.8, 854.1)`
- **Left Hand Box:** `(408.2, 421.0) -> (832.0, 854.1)`

## 3. Tag Policy & Crop Artifact Elimination (Phase 2)

- **Input Candidates:** `tube top`, `cleavage`, `white thighhighs`, `barefoot`, `sunglasses`, `close-up`, `portrait`, `solo focus`, `breast focus`, `head out of frame`, `general`, `hatsune miku`
- **Retained after Policy:** `tube top, cleavage, white thighhighs, barefoot, sunglasses`
- **Blacklisted Artifacts Blocked:** `close-up`, `portrait`, `solo focus`, `breast focus`, `head out of frame` (Count: 0 leaked)
- **Rating & Character Leaks Blocked:** `general`, `hatsune miku` (Count: 0 leaked)

## 4. Layer-Based Conflict Resolution (Phase 4)

| Conflict Scenario | Input Tags | Result | Verification |
| :--- | :--- | :--- | :--- |
| **Coexistence** | `tube top` + `cleavage` | Retained both: `tube top, cleavage, white thighhighs, barefoot` | `cleavage` NOT occluded by `tube top` |
| **Occlusion** | `turtleneck` + `cleavage` | Cleavage suppressed: `turtleneck, white thighhighs, barefoot` | `turtleneck` correctly occludes `cleavage` |
| **Footwear / Barefoot** | `boots` + `thighhighs` + `shoes` + `barefoot` | Barefoot dropped: `boots, thighhighs, shoes` | `boots` coexists with `thighhighs`; `shoes` suppresses `barefoot` |

## 5. ControlNet OpenPose JSON Export (Phase 5)

```json
{
  "version": 1.3,
  "people": [
    {
      "person_id": [
        -1
      ],
      "pose_keypoints_2d": [
        417.5,
        156.35999999999999,
        0.95,
        417.5,
        252.60000000000002,
        0.95,
        235.12,
        276.66,
        0.95,
        185.37999999999997,
        469.14,
        0.95,
        210.25,
        637.5600000000001,
        0.95,
        599.88,
        276.66,
        0.95,
        649.62,
        469.14,
        0.95,
        624.75,
       
  ...
}
```

## 6. Synthesized Output

### Scene & Shared Framing (Pass 1):
> masterpiece, best quality, close-up, depth of field, holding, sitting, couch, red couch, indoors, bedroom, bookshelf, window, soft lighting, cleavage, holding spoon, lamp

### Character Wardrobe & Details (Pass 2):
> 2girls, blonde hair, braid, blue eyes, shorts, green bracelet, earrings, open shirt, white shirt, dress shirt, skirt, pink skirt, black bralette, lace trim, keyhole, sideboob, cardigan around waist, pink cardigan, eating, potato chips, chips (food), cups, straws, butterfly hair ornament, hair ornament, hair clip, pink nails, green nails, blue nails, multicolored nails, leg warmers, bare legs, crossed legs, potted plant, flower, mirror, picture frame, smile, looking at viewer

### Pipeline Status:
- Warnings: `['Gemma output unusable; used the mechanical merge']`
- Dropped Layer Conflicts: `[]`
