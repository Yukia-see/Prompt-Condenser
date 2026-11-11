class AntigravityTagFilter:
    def __init__(self):
        self.wide_shots = {"full body", "cowboy shot", "wide shot"}
        self.close_shots = {"close-up", "portrait", "upper body", "face focus"}
        self.tops = {"bra_top", "shirt", "jacket", "tank top", "camisole", "bikini top", "tube top", "crop top", "sweater", "hoodie"}
        self.bottoms = {"pants", "skirt", "shorts", "bikini bottoms", "panties", "underwear", "dress", "jeans", "leggings"}
        self.static_poses = {"standing", "sitting", "kneeling"}
        self.limb_anchors = {"hand", "hands", "arm", "arms", "holding", "reaching", "pulling", "adjusting", "crossed", "lap", "hips"}
        self.archetype_overrides = {"bunny girl", "cat girl", "fox girl"}
        self.uniforms = {"maid", "maid outfit", "plaid dress", "school uniform", "serafuku", "suit"}

    def process_tags(self, raw_tag_string: str) -> str:
        """Sanitizes scanner tags and injects structural audit flags for the LLM pass."""
        # 1. Normalize and strip inline weights/parentheses
        raw_list = [t.strip().lower() for t in raw_tag_string.split(",") if t.strip()]
        tags = []
        for tag in raw_list:
            clean = tag.replace("(", "").replace(")", "").split(":")[0].strip()
            if clean and clean not in tags:
                tags.append(clean)

        # 2. Implicit Full-Body Inference (Footwear/Legwear Trigger)
        if any(fw in t for t in tags for fw in ["thighhighs", "mary janes", "boots", "heels", "barefoot", "sandals"]):
            if not any(cam in tags for cam in self.wide_shots | self.close_shots):
                tags.insert(0, "full body")

        # 3. Camera Paradox Resolution (Wide overrides Close)
        has_wide = any(cam in tags for cam in self.wide_shots)
        if has_wide:
            tags = [t for t in tags if t not in self.close_shots]

        # 4. Archetype vs. Uniform Collision Purge
        if any(u in t for t in tags for u in self.uniforms):
            tags = [t for t in tags if t not in self.archetype_overrides]

        # 5. Bottomless Hazard Flagging
        has_top = any(any(top in t for top in self.tops) for t in tags)
        has_bottom = any(any(bot in t for bot in self.bottoms) for t in tags)
        if has_wide and has_top and not has_bottom:
            tags.append("MISSING_BOTTOMS_FLAG")

        # 6. Unrouted Limb Flagging
        has_static = any(pose in tags for pose in self.static_poses)
        has_limb_anchor = any(any(limb in t for limb in self.limb_anchors) for t in tags)
        if has_static and not has_limb_anchor:
            tags.append("UNANCHORED_LIMBS_FLAG")

        return ", ".join(tags)