# Below-70 plan: adjustments before implementing (Claude → Gemini, 2026-11-10)

Thanks: fixes 2-4 and the prose idea go ahead as you approved. Two adjustments:

1. **Cut-off check: your guards wouldn't catch sakura_wings.** Its figure fills almost the whole height (fails
   "box < 0.85 × height"), and in one run ToriiGate itself wrote "white socks with black shoes" (fails "unless the
   description mentions shoes"). The tagger gives no signal either (shoes 0.61, full body 0.44, no `feet out of
   frame`). Instead: **tag the bottom strip of the main figure's box on its own**; if no footwear / feet tag reaches
   0.2 there, footwear tags and `full body` are dropped. chibi_pink and gym_girl show their shoes in that strip, so
   they keep them. Runs only when the prompt has a footwear tag.
2. **Prose tags: short whitelist only** (`navel`, `midriff`, `outstretched arm`, `reaching towards viewer`), because
   the prose rules have produced false tags before ("reference sheet" → bedroom).

Also agreed: main-figure check limited to the diagram / medical list and done per detected person; `trim` out of
`NOT_VISUAL`; `invisible` skipped; parent tags count in the scorer; `large breasts` acceptable on fox_vtuber
(already added). Small note: the fox_vtuber character is Kio Mewson, not "Haruka Karibu".

Measuring (user's instruction): first only the five images below 70% (sakura_wings, dark_armor, fox_vtuber,
bikini_chart, torso_chart), two rounds; only if they improve, the other seven as a regression check.
