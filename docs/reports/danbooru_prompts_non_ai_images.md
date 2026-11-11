# Canonical Danbooru Prompts for Non-AI Project Images

This document establishes the verified, canonical Danbooru ground-truth prompts for all **non-AI image files** in the Prompt Condensor project.

Every image has been visually inspected, cross-referenced with upstream Danbooru taxonomy conventions, and analyzed against SmilingWolf's WD EVA02-Large v3 tagger outputs.

---

## Catalog of Non-AI Images

| Image File | Location | Resolution | Subject / Franchise | Rating |
| :--- | :--- | :--- | :--- | :--- |
| **`beret_girl.jpg`** | `tests/eval_local/` | 649 × 900 | Nishizumi Miho (Girls und Panzer - Anzio Uniform) | General |
| **`bikini_chart.jpg`** | `tests/eval_local/` | 1312 × 1199 | Bikini anatomy study / reference chart | Sensitive |
| **`chibi_pink.jpg`** | `tests/eval_local/` | 637 × 900 | High-angle dynamic foreshortened serafuku girl | Sensitive |
| **`dark_armor.jpg`** | `tests/eval_local/` | 877 × 900 | Gothic fantasy dark knight / vampire noblewoman | Sensitive |
| **`fox_vtuber.png`** | `tests/eval_local/` | 407 × 1009 | Mewson (Green fox kemonomimi VTuber) | Sensitive |
| **`gym_girl.jpg`** | `tests/eval_local/` | 506 × 900 | Brown twintail girl in buruma on one leg | Sensitive |
| **`office_two_girls.jpg`** | `tests/eval_local/` | 832 × 1216 | Suzukaze Aoba & Lillie in office (Fisheye) | Sensitive |
| **`rouge_bat.jpg`** | `tests/eval_local/` | 1470 × 1130 | Rouge the Bat in mirror reflection | Sensitive |
| **`sakura_wings.jpg`** | `tests/eval_local/` | 637 × 900 | Kinomoto Sakura (OP1 "Catch You Catch Me" battle costume) | General |
| **`schoolgirl_two_panels.jpg`** | `tests/eval_local/` | 1283 × 1008 | Gyaru schoolgirl 2-panel comic (Pocky kiss / eating) | Sensitive |
| **`ruri_rocks.jpg`** | `tests/eval_local/fresh/` | 1000 × 956 | 3 Kogal / gyaru schoolgirls (Ruri Rocks) | Questionable |
| **`torso_chart.jpg`** | `tests/eval_local/fresh/` | 1024 × 1536 | Torso / anatomy reference lineart mannequin | Sensitive |
| **`HQwyhBIbAAAguCD.jpg`** | Root & `eval_dataset/` | 653 × 680 | Pyra, Mythra, Pneuma in swimsuits (Xenoblade) | Sensitive |
| **`HVvgeziXsAA6Crn.png`** | `tests/eval_dataset/` | 707 × 315 | Artist credit banner card (Signature / Blue card) | General |
| **`Mewson-2.png`** | `tests/eval_dataset/` | 342 × 1003 | Mewson VTuber full-body character sheet | Sensitive |
| **`character.jpg`** | `packaging/arch/` | 1264 × 1568 | Brunette woman with sunglasses on bed | Questionable |
| **`scene.jpg`** | `packaging/arch/` | 1472 × 2176 | Blonde eyepatch schoolgirl adjusting white thighhigh | Sensitive |

> [!NOTE]
> `195346 nonscaled_00001_.png` (contains ComfyUI `tpgANIMAMessymix` workflow metadata) and `HWP1dffawAA5Y7K.jpg` (`two_blondes.jpg`) have been confirmed as AI-generated and are intentionally excluded from the non-AI benchmark set.

---

## 1. Primary Evaluation Suite (`tests/eval_local/`)

### 1.1 `beret_girl.jpg`
- **Franchise / Character**: *Girls und Panzer* — Nishizumi Miho wearing the Anzio High School uniform.
- **Composition**: Tilted high angle, clenched fist raised, frowning with light blush, white background.
- **Canonical Danbooru Prompt**:
```danbooru
nishizumi miho, girls und panzer, 1girl, solo, short hair, brown hair, brown eyes, light frown, closed mouth, blush, beret, black beret, tilted headwear, anzio school uniform, school uniform, white shirt, collared shirt, wing collar, long sleeves, black necktie, necktie, black skirt, pleated skirt, miniskirt, clenched hand, clenched fist, upper body, foreshortening, looking at viewer, white background, simple background, character name, japanese text
```
- **Comparison to `beret_girl.txt`**:
  - *Added canonical tags*: `nishizumi miho`, `girls und panzer`, `anzio school uniform`, `pleated skirt`, `miniskirt`, `light frown`, `tilted headwear`.
  - *Existing answer file retains*: 100% compatibility while missing character/school specifics.

---

### 1.2 `bikini_chart.jpg`
- **Subject**: Anatomy and costume reference sheet detailing the fit, side ties, and curvature of a black bikini on a female torso from front, three-quarter, and side views.
- **Composition**: Multiple views, head and lower legs out of frame, instructional indicator arrows.
- **Canonical Danbooru Prompt**:
```danbooru
1girl, multiple views, head out of frame, cropped arms, cropped legs, cropped torso, swimsuit, bikini, black bikini, string bikini, side-tie bikini bottom, medium breasts, breasts, cleavage, navel, stomach, collarbone, toned, abs, skindentation, how to, reference sheet, arrow (symbol), simple background, white background
```
- **Comparison to `bikini_chart.txt`**:
  - *Added canonical tags*: `swimsuit`, `string bikini`, `medium breasts`, `stomach`, `cropped arms`, `cropped legs`, `cropped torso`, `abs`, `toned`, `skindentation`, `how to`.
  - *Existing answer file*: Accurately captured core geometry, but omitted standard anatomy descriptors (`stomach`, `medium breasts`, `abs`).

---

### 1.3 `chibi_pink.jpg`
- **Subject**: Anime schoolgirl in dynamic high-angle perspective with extreme foreshortening, clenched fists raised cheering, cropped serafuku revealing underboob and midriff, dark green pleated miniskirt, black pantyhose, and uwabaki indoor shoes.
- **Composition**: Full-body downward perspective. **Not a true chibi** (proportions are stylized standard anime with defined breast anatomy and underboob, not super-deformed 2-head-tall SD).
- **Canonical Danbooru Prompt**:
```danbooru
1girl, solo, long hair, pink hair, ahoge, blue eyes, closed mouth, blush, serafuku, sailor collar, cropped shirt, white shirt, short sleeves, underboob, bare midriff, navel, pleated skirt, miniskirt, dark green skirt, pantyhose, black pantyhose, uwabaki, shoes, white footwear, clenched hands, hands up, raised fists, foreshortening, dynamic angle, looking at viewer, full body, simple background, white background
```
- **Comparison to `chibi_pink.txt`**:
  > [!IMPORTANT]
  > Existing `chibi_pink.txt` contained severe errors:
  > 1. Called her outfit `black pants` (she is wearing a **dark green pleated miniskirt** over **black pantyhose**).
  > 2. Labeled as `chibi` (she has full anatomical definition, underboob, and standard proportions foreshortened).
  > 3. Missing `underboob`, `pantyhose`, `black pantyhose`, and `uwabaki`.

---

### 1.4 `dark_armor.jpg`
- **Subject**: Dark fantasy gothic warrior / vampire noblewoman in black armored dress with emerald glowing embroidery, spiky pauldrons, gauntlets, and dagger, holding a dripping chalice of blood.
- **Composition**: Dual presentation (full-color rendered character portrait on left, grayscale process sketches on right), dark vignette background.
- **Canonical Danbooru Prompt**:
```danbooru
1girl, solo, black hair, very long hair, hair over one eye, green eyes, red lips, lipstick, smile, armor, shoulder armor, spikes, gauntlets, armlet, black dress, long dress, wide sleeves, cleavage cutout, green trim, side slit, thighhighs, black thighhighs, tassel, headdress, hair ornament, holding, holding weapon, weapon, knife, dagger, holding dagger, holding knife, chalice, cup, holding cup, blood, dripping, bone, skeleton, large breasts, cleavage, multiple views, reference sheet, english text, black background, dark background
```
- **Comparison to `dark_armor.txt`**:
  - *Added canonical tags*: `green eyes`, `dagger`, `holding dagger`, `holding weapon`, `cleavage cutout`, `armlet`, `bone`, `skeleton`.
  - *Resolution of grayscale bug*: Visual inspection confirmed the main subject is in **full color** (deep greens, crimson lips, red blood, metallic sheen); monochrome detection must not trigger on the tutorial side panels.

---

### 1.5 `fox_vtuber.png`
- **Franchise / Character**: Independent VTuber **Mewson** (`virtual youtuber`).
- **Subject**: Kemonomimi fox girl with vibrant green hair, emerald/aqua eyes, large fluffy fox ears and matching giant fox tail.
- **Outfit**: Off-shoulder ribbed cardigan, black choker, frilled halterneck bra, green low-rise panties, sheer black thighhighs, painted green nails.
- **Canonical Danbooru Prompt**:
```danbooru
1girl, solo, virtual youtuber, fox girl, kemonomimi, green hair, long hair, hair between eyes, aqua eyes, green eyes, fox ears, animal ears, animal ear fluff, fox tail, green tail, huge tail, hair flower, hair ornament, choker, black choker, large breasts, huge breasts, cleavage, cleavage cutout, off shoulder, halterneck, frills, cardigan, long sleeves, navel, bare shoulders, green panties, underwear, black thighhighs, thighhighs, green nails, nail polish, standing, straight-on, full body, tachi-e, looking at viewer, smile, simple background, transparent background
```
- **Comparison to `fox_vtuber.txt`**:
  - *Added canonical tags*: `virtual youtuber`, `green tail`, `huge tail`, `green eyes`, `straight-on`, `tachi-e`, `green nails`, `nail polish`, `bare shoulders`.

---

### 1.6 `gym_girl.jpg`
- **Subject**: Energetic anime girl in Japanese gym uniform with oversized breasts, standing on one foot with the other knee bent backward, leaning forward with raised clenched fists.
- **Hair/Face**: Brown twintails with black ribbons, ahoge, aqua eyes, wide joyful open grin showing teeth, pink cheek blush.
- **Outfit**: White gym t-shirt stretched tightly over gigantic breasts, classic dark blue buruma (gym shorts), white crew socks, white sneakers.
- **Canonical Danbooru Prompt**:
```danbooru
1girl, solo, brown hair, twintails, ahoge, aqua eyes, hair ribbon, black ribbon, smile, grin, open mouth, teeth, blush, looking at viewer, gigantic breasts, huge breasts, cleavage, gym uniform, gym shirt, white shirt, short sleeves, buruma, gym shorts, blue shorts, white socks, socks, sneakers, white footwear, shoes, clenched hands, raised fists, standing on one leg, leg up, bent over, leaning forward, full body, pink background, simple background, white outline
```
- **Comparison to `gym_girl.txt`**:
  - *Added canonical tags*: `open mouth`, `buruma`, `gym shirt`, `sneakers`, `white footwear`, `bent over`, `leaning forward`, `full body`.

---

### 1.7 `office_two_girls.jpg`
- **Franchises / Characters**:
  1. **Suzukaze Aoba** (*New Game!*): Purple twintails, hair flower, blue skirt suit blazer with pink neck ribbon.
  2. **Lillie** (*Pokémon Sun/Moon*): Blonde braided ponytail, blunt bangs, white sailor collar blouse, white pleated skirt, holding standard Poké Ball.
- **Composition**: Fisheye high-angle shot in modern game development studio / office with computer workstations, multiple monitors, keyboards, sticky notes, and whiteboard.
- **Canonical Danbooru Prompt**:
```danbooru
2girls, multiple girls, suzukaze aoba, lillie (pokemon), suzukaze aoba (cosplay), purple hair, twintails, blunt bangs, hair flower, hair ornament, purple eyes, suit, skirt suit, blue jacket, blazer, formal, pink ribbon, neck ribbon, blue skirt, blonde hair, long hair, braid, ponytail, green eyes, white shirt, collared shirt, sailor collar, white skirt, pleated skirt, miniskirt, backpack, poke ball, poke ball (basic), large breasts, open mouth, smile, :d, clenched hands, standing on one leg, leg up, sneakers, white socks, socks, full body, from above, fisheye, breast press, indoors, office, desk, computer, monitor, keyboard (computer), mouse (computer), sticky note, whiteboard, tile floor, tiles, potted plant
```
- **Comparison to `office_two_girls.txt`**:
  - *Added canonical tags*: `blunt bangs`, `poke ball (basic)`, `formal`, `keyboard (computer)`, `mouse (computer)`, `whiteboard`, `full body`.
  - Captures the exact identities of both characters and the detailed office environment.

---

### 1.8 `rouge_bat.jpg`
- **Franchise / Character**: *Sonic the Hedgehog* — **Rouge the Bat**.
- **Subject**: Female anthropomorphic bat in black skin-tight strapless bodysuit with pink heart breastplate, elbow-length white opera gloves, inspecting her reflection in a full-length dressing mirror.
- **Canonical Danbooru Prompt**:
```danbooru
rouge the bat, sonic (series), 1girl, solo, anthro, furry, furry female, bat girl, bat wings, black wings, wings, bat ears, animal ears, white fur, body fur, animal nose, aqua eyes, blue eyes, eyeshadow, blue eyeshadow, pink lips, lipstick, eyelashes, large breasts, breasts, cleavage, strapless, bodysuit, black bodysuit, heart, pink heart, gloves, white gloves, elbow gloves, hand on own hip, half-closed eyes, closed mouth, looking down, looking at mirror, reflection, mirror, cowboy shot, standing, indoors
```
- **Comparison to `rouge_bat.txt`**:
  - *Added canonical tags*: `sonic (series)`, `anthro`, `bat ears`, `black wings`, `pink heart`, `hand on own hip`, `cowboy shot`, `closed mouth`.

---

### 1.9 `sakura_wings.jpg`
- **Franchise / Character**: *Cardcaptor Sakura* — **Kinomoto Sakura** in the iconic Season 1 Opening 1 Battle Costume ("Catch You Catch Me").
- **Costume**: White baker boy cap with red cherry bobbles, cropped sailor top with deep blue sailor collar and gold buttons, white ruffled miniskirt, black and white striped thighhighs, red mary janes, feathered angel wings.
- **Pose**: Outstretched hand reaching forward toward the viewer with strong perspective foreshortening.
- **Canonical Danbooru Prompt**:
```danbooru
kinomoto sakura, cardcaptor sakura, 1girl, solo, short hair, brown hair, ahoge, antenna hair, green eyes, open mouth, :o, looking at viewer, hat, white headwear, baker boy cap, hair ornament, hair bobbles, cherry hair ornament, wings, white wings, angel wings, serafuku, sailor collar, blue sailor collar, blue jacket, long sleeves, cropped shirt, bare midriff, navel, white skirt, pleated skirt, frilled skirt, miniskirt, thighhighs, striped thighhighs, black and white striped thighhighs, mary janes, red footwear, shoes, outstretched arm, outstretched hand, reaching towards viewer, foreshortening, full body, pink background, heart, cherry blossoms
```
- **Comparison to `sakura_wings.txt`**:
  - *Added canonical tags*: `cardcaptor sakura`, `antenna hair`, `baker boy cap`, `cherry hair ornament`, `angel wings`, `blue sailor collar`, `cropped shirt`, `mary janes`, `full body`.

---

### 1.10 `schoolgirl_two_panels.jpg`
- **Subject**: 2-panel comic illustration of a trendy gyaru / kogal high school girl.
  - *Left Panel*: Seated on classroom chair at desk with legs crossed, leaning back, holding a chocolate Pocky stick between her teeth, tongue playfully out.
  - *Right Panel*: Standing up, leaning forward toward viewer over desk, making a peace sign (V sign) with a bright beaming smile.
- **Outfit**: Brown knitted cardigan over unbuttoned white blouse, loose diagonal-striped blue and red neckties, navy plaid pleated miniskirt, slouchy loose white socks, penny loafers.
- **Canonical Danbooru Prompt**:
```danbooru
1girl, multiple views, 2koma, gyaru, kogal, long hair, black hair, ponytail, scrunchie, purple scrunchie, blue eyes, earrings, hoop earrings, necklace, heart necklace, bracelet, nail polish, pink nails, blush, tongue, tongue out, licking lips, :q, smile, open mouth, v, peace sign, school uniform, white shirt, collared shirt, open shirt, cleavage, large breasts, bowtie, blue bowtie, striped bowtie, necktie, red necktie, striped necktie, loose necktie, cardigan, brown cardigan, plaid skirt, pleated skirt, miniskirt, blue skirt, loose socks, white socks, loafers, shoes, pocky, food, eating, holding food, holding, school bag, bag, sitting, chair, crossed legs, standing, leaning forward, school desk, desk, classroom, indoor, grid background, pink background
```
- **Comparison to `schoolgirl_two_panels.txt`**:
  - *Added canonical tags*: `2koma`, `kogal`, `open shirt`, `peace sign`, `loafers`, `classroom`, `indoor`.
  - Accurately captures both panels in Danbooru narrative style without confusing them as two separate girls.

---

## 2. Fresh Non-AI Test Suite (`tests/eval_local/fresh/`)

### 2.1 `ruri_rocks.jpg` (identical to `HWL8f8wbYAAMJ-a.jpg`)
- **Subject**: 3 Gyaru / Kogal schoolgirls walking together.
  - *Left*: Busty girl in white blouse and blue vest with high ponytail and button-adorned backpack, flashing a peace sign.
  - *Center (Back View)*: Girl viewed from behind in extreme micro-miniskirt showing underbutt and thong, looking back over shoulder with a flirtatious wink.
  - *Right*: Blonde girl in loose socks smiling alongside.
- **Canonical Danbooru Prompt**:
```danbooru
3girls, multiple girls, gyaru, kogal, full body, standing, black hair, long hair, high ponytail, scrunchie, hair scrunchie, brown eyes, earrings, brown hair, short hair, blonde hair, two-tone hair, hairclip, purple eyes, blue eyes, huge breasts, gigantic breasts, cleavage, oversized shirt, white shirt, blue vest, collared shirt, plaid skirt, pleated skirt, miniskirt, microskirt, underbutt, thong, from behind, looking back, looking at viewer, smile, open mouth, wink, v, peace sign, thick thighs, loose socks, white socks, loafers, brown footwear, backpack, school bag, button badge, badge, character name, english text, yellow background, white vertical stripes, striped background, white outline
```

---

### 2.2 `torso_chart.jpg`
- **Subject**: Anatomical drawing reference sheet illustrating female torso proportions, breast volume, hip curvature, and glute angles across front, 3/4, side, and rear perspectives.
- **Canonical Danbooru Prompt**:
```danbooru
no humans, mannequin, headless, multiple views, reference sheet, how to, sketch, lineart, monochrome, torso, breasts, large breasts, cleavage, ass, hips, waist, nude, from side, from behind, profile, cropped arms, cropped legs, simple background, white background
```

---

## 3. Dataset & Project Non-AI Images

### 3.1 `HQwyhBIbAAAguCD.jpg` (Root & `tests/eval_dataset/`)
- **Franchise / Characters**: *Xenoblade Chronicles 2* — The Aegis trio in beach swimwear:
  1. **Pyra** (`pyra (xenoblade)`): Crimson short bob, red eyes, red floral hair clip, black strappy bikini, chest core crystal.
  2. **Mythra** (`mythra (xenoblade)`): Long golden blonde hair, yellow eyes, ornate tiara, bright yellow bikini, chest core crystal.
  3. **Pneuma** (`pneuma (xenoblade)`): Emerald green hair in high ponytail, green eyes, white bikini with thigh strap, chest core crystal.
- **Canonical Danbooru Prompt**:
```danbooru
3girls, multiple girls, pyra (xenoblade), mythra (xenoblade), pneuma (xenoblade), xenoblade chronicles 2, xenoblade (series), red hair, bob cut, red eyes, red flower, hair flower, blonde hair, very long hair, swept bangs, yellow eyes, green hair, aqua hair, high ponytail, green eyes, tiara, headpiece, jewelry, gem, chest jewel, core crystal, swimsuit, bikini, black bikini, yellow bikini, white bikini, mismatched bikini, two-tone bikini, strap, thigh strap, skindentation, large breasts, huge breasts, cleavage, underboob, sideboob, bare shoulders, bare legs, stomach, navel, thick thighs, wide hips, curvy, ass, barefoot, toes, feet, standing, smile, blush, looking at viewer, open mouth, full body, white background, simple background
```

---

### 3.2 `HVvgeziXsAA6Crn.png` (`tests/eval_dataset/`)
- **Subject**: Graphic design credit card / social banner card featuring artist handle, Twitter username, and Pixiv ID in clean white typography on a blue background.
- **Canonical Danbooru Prompt**:
```danbooru
no humans, text focus, english text, artist name, twitter username, pixiv id, signature, dated, monochrome, blue theme, blue background, simple background
```

---

### 3.3 `Mewson-2.png` (`tests/eval_dataset/`)
- **Franchise / Character**: Mewson (`virtual youtuber`).
- **Subject**: Full-length standing character concept art of the green fox VTuber in underwear.
- **Canonical Danbooru Prompt**:
```danbooru
1girl, solo, virtual youtuber, fox girl, kemonomimi, green hair, very long hair, hair between eyes, green eyes, fox ears, animal ears, animal ear fluff, green tail, fox tail, huge tail, hair flower, hair ornament, large breasts, huge breasts, cleavage, navel, bare shoulders, underwear, panties, green panties, black panties, thighhighs, black thighhighs, green nails, blue nails, multicolored nails, nail polish, standing, full body, straight-on, tachi-e, looking at viewer, smile, simple background, transparent background
```

---

### 3.4 `packaging/arch/character.jpg`
- **Subject**: Curvy brunette woman with sunglasses pushed up onto her dark hair, relaxing on an unmade bed in high-leg blue panties and spaghetti-strap top, resting her hands on her raised knees.
- **Canonical Danbooru Prompt**:
```danbooru
1girl, solo, mature female, long hair, black hair, messy hair, grey eyes, blue eyes, sunglasses, sunglasses on head, eyewear on head, thick lips, lips, freckles, gigantic breasts, huge breasts, cleavage, armpits, arms up, navel, curvy, realistic, camisole, spaghetti strap, light blue tube top, underwear, panties, highleg panties, thong, blue panties, sitting, on bed, bed, knees up, hands on knees, curtains, indoors, looking at viewer, closed mouth
```

---

### 3.5 `packaging/arch/scene.jpg`
- **Subject**: Blonde schoolgirl with twin low twintails, ahoge, and a white eyepatch over one eye, wearing a navy school swimsuit underneath an off-the-shoulder white sailor shirt, seated on the floor adjusting her white thighhigh stocking with an O-ring garter strap.
- **Canonical Danbooru Prompt**:
```danbooru
1girl, solo, blonde hair, very long hair, long hair, low twintails, twintails, hair bun, ahoge, pom pom hair ornament, yellow eyes, eyepatch, one eye covered, parted lips, blush, school swimsuit, one-piece swimsuit, blue one-piece swimsuit, swimsuit under clothes, serafuku, white shirt, long sleeves, off shoulder, single bare shoulder, sailor collar, blue sailor collar, bowtie, red bow, red bowtie, pom pom (clothes), choker, white choker, thighhighs, white thighhighs, thigh strap, o-ring thigh strap, garter straps, skindentation, no shoes, bare feet, toes, sitting, knee up, adjusting legwear, adjusting clothes, thick thighs, mole, mole on thigh, collarbone, border, white border, blue background
```

---

## 4. Key Tagging Standards & Lessons Learned

1. **Danbooru Tag Precedence Order**:
   - `[Character / Franchise]` (e.g. `kinomoto sakura, cardcaptor sakura`)
   - `[Subject / Count]` (e.g. `1girl, solo` or `2girls, multiple girls`)
   - `[Anatomy & Physical Traits]` (hair color, hair length, eye color, breast size, ears/tail)
   - `[Clothing & Layers]` (top, collar, skirt/bottoms, underwear, legwear, footwear)
   - `[Accessories & Headwear]` (hats, berets, chokers, jewelry, weapons, props)
   - `[Pose, Expression & Action]` (clenched hands, peace sign, eating, looking at viewer)
   - `[Setting, Environment & Background]` (indoors, classroom, office, simple background)
   - `[Framing & Meta / Style]` (full body, cowboy shot, fisheye, multiple views, 2koma)

2. **Ground Truth Corrections Applied**:
   - `chibi_pink.jpg`: Corrected `black pants` to `dark green pleated miniskirt` + `black pantyhose` + `uwabaki`. Removed erroneous `chibi` tag.
   - `dark_armor.jpg`: Recognized full-color gothic palette with green glowing tassels and crimson details; prevented monochrome misclassification.
   - `sakura_wings.jpg`: Added exact franchise tags (`kinomoto sakura`, `cardcaptor sakura`), `baker boy cap`, and `cherry hair ornament`.
   - `beret_girl.jpg`: Added character tags (`nishizumi miho`, `anzio school uniform`, `girls und panzer`).
