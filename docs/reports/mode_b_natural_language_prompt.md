# Mode B: Natural Language & Hybrid Prompt (Cleaned)

**Adjustment**: Removed `bursting breasts` and `cleavage` (which conflicted with the closed neckline of the printed tube top and triggered the Portrait gravure/Seiza bias), and locked in forward-leg geometry (`leaning back`, `one knee up`, `legs crossed`, `foreshortening`).

---

## Option 1: Pure Natural Language Version (Anima / Qwen / Flux / SDXL)

```text
masterpiece, best quality. A young woman with a full, curvy figure, slim waist, and wide hips sits leaning back in a softly lit room, captured from a dynamic low angle with shallow depth of field and natural foreshortening. She has long, messy black hair, large grey eyes with long eyelashes, a gentle head tilt, and sunglasses perched on top of her head. She wears a form-fitting, taut light blue tube top with printed text across the front and matching light blue bottoms. With her legs crossed and one knee raised in front of her, she rests her hands on her knees to adjust her white thigh-high stockings and white shoes while looking at the viewer.
```

---

## Option 2: Studio Backdrop Version (Matches `scene.jpg` Polka-Dot Studio Scene)

```text
masterpiece, best quality. Against a simple light blue and aqua polka-dot background with soft, serene lighting and a low camera angle looking up, a curvy young woman with a full figure, slim waist, and wide hips sits in a relaxed posture with her legs crossed and one knee raised toward the viewer. She has long, messy black hair, large grey eyes, and sunglasses resting on her head, wearing a taut light blue printed tube top and matching light blue bottoms. Her hands rest on her elevated knees as she adjusts her white thigh-high stockings with garter straps and white shoes.
```

---

## Option 3: Hybrid Format (Scene Prose + Cleaned Mode B Tags)

### `[Scene / Environment & Shared Framing]`:
```text
masterpiece, best quality, simple background, blue background, aqua background, polka dot background, soft lighting, serene atmosphere, low angle, camera looking up, foreshortening, A curvy young woman with long messy black hair and sunglasses on her head sits in a relaxed pose with her legs crossed and one knee raised forward, wearing a taut light blue printed tube top and matching bottoms while resting her hands on her knees to adjust her white thigh-high stockings.
```

### `[Character 1]`:
```text
1girl, solo, long hair, black hair, messy hair, large eyes, grey eyes, long eyelashes, sunglasses on head, eyewear on head, curvy, large breasts, slim waist, wide hips, light blue tube top, print shirt, text on clothes, taut clothes, light blue bottoms, white thighhighs, garter straps, o-ring thigh strap, white socks, white shoes, sitting, relaxed, legs crossed, one knee up, hands on knees, adjusting legwear, head tilt, looking at viewer
```

### `[Recommended Negative Prompt (Portrait Anti-Seiza Lock)]`:
```text
worst quality, low quality, (seiza:1.3), (kneeling:1.3), (sitting on heels:1.3), wariza, bursting breasts, cleavage, bare foot
```
