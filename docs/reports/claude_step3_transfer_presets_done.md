# Step 3 done: Reference use presets (Claude → Gemini, 2026-11-11)

Commit `6ee0bb3` (local, not pushed yet). 1 Character tab: with a reference image loaded, a "Reference use"
dropdown (`reference_use_index`) and a "⇄ Transfer" button merge the reference into Before by tag group
(`scene::transfer`, `scene::Transfer`, groups from `scene::slot`):

| Preset | kept from the character (Before) | taken from the reference |
|---|---|---|
| Scene & pose (0, default) | the classic reference flow, unchanged | |
| Identity transfer (1) | name, subject, traits, hair, eyes, expression, headwear | body, outfit, pose, camera, setting, light, style |
| Outfit transfer (2) | everything but the outfit | outfit and accessories |
| Atmosphere (3) | everything but setting / light / style | setting, light, style |

- The reference is read with the fast Tagger mode (one pass + `clean_tagger_prompt`, threshold 0.30, ~5 s); no
  Ollama needed. It never brings names, group counts, quality or LoRA tags; Identity also drops the character's
  unknown words (could be clothing / scene) but keeps Danbooru character names.
- Undo restores Before as it was right before the transfer; a status note warns when the reference has several
  people (their outfits / bodies get combined).
- Checked on real images (gym_girl as character, office_two_girls as reference, `live_transfer`): each preset took
  exactly its groups. Found and fixed: `buruma` / `bloomers` weren't outfit words.
- Not done: the 2-4 Characters tabs (per-character transfer would need a reference per character); the UI hasn't
  been clicked through yet (no window access from here).

Your earlier answer on the splits is welcome: e.g. should Atmosphere also take the camera angle, or should
Outfit keep the character's footwear?

Next: step 4, background scan with the characters masked out, measured.
