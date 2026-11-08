# Live Pipeline Test Report: Phases 0-5 End-to-End Validation

- **Image Tested:** `packaging/arch/character.jpg` (1264x1568)
- **Detection Figure:** `(10.0, 3.0)` to `(1259.0, 1567.0)`
- **Figure Height:** `1564.0px`

## 1. Pure Zone Decomposition (Phase 2)

| Zone | Geometry Box (x0, y0, x1, y1) | Target Anatomical Features |
| :--- | :--- | :--- |
| **Upper** | `(0.0, 0.0) -> (1264.0, 925.8)` | Upper figure details |
| **Middle** | `(0.0, 409.6) -> (1264.0, 1160.4)` | Middle figure details |
| **Lower** | `(0.0, 800.6) -> (1264.0, 1568.0)` | Lower figure details |

## 2. Wrist-Centered Hand Bounding Boxes (Phase 5)

- **Proportional Hand Radius:** `281.5px` ($18\% \times H_\text{figure}$)
- **Right Hand Box:** `(40.7, 534.8) -> (603.8, 1097.8)`
- **Left Hand Box:** `(665.2, 534.8) -> (1228.3, 1097.8)`

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
        634.5,
        190.68,
        0.95,
        634.5,
        315.8,
        0.95,
        359.71999999999997,
        347.08,
        0.95,
        284.78,
        597.32,
        0.95,
        322.25,
        816.28,
        0.95,
        909.28,
        347.08,
        0.95,
        984.22,
        597.32,
        0.95,
        946.75,
        816.28,
        0.95,
        634.5
  ...
}
```

## 6. Synthesized Output

### Scene & Shared Framing (Pass 1):
> masterpiece, best quality, full body, depth of field, lying, bed, cleavage, sitting on bed

### Character Wardrobe & Details (Pass 2):
> 1girl, solo, black hair, smile, bikini, long hair, sunglasses on head, tube top, light blue bikini bottoms, white thighhighs, barefoot, looking at viewer

### Pipeline Status:
- Warnings: `['Gemma output unusable; used the mechanical merge']`
- Dropped Layer Conflicts: `[]`
