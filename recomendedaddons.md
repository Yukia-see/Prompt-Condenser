{
  "schema_version": "3.0.0",
  "description": "Master Danbooru Prompt Conflict, Lock-In, and Synergy Engine",
  "1_wd14_scrubber_and_deduplication": {
    "always_strip_junk": [
      "no_humans",
      "commentary_request",
      "tagme",
      "bad_id",
      "translated",
      "signature",
      "watermark",
      "artist_name",
      "patreon_logo",
      "twitter_username",
      "web_address",
      "dated",
      "text",
      "english_text",
      "japanese_text",
      "speech_bubble",
      "comic",
      "border",
      "white_border",
      "frame"
    ],
    "specificity_hierarchy": {
      "hair_length": [
        "absurdly_long_hair",
        "very_long_hair",
        "long_hair",
        "medium_hair",
        "short_hair",
        "very_short_hair"
      ],
      "breast_size": [
        "gigantic_breasts",
        "hyper_breasts",
        "huge_breasts",
        "large_breasts",
        "medium_breasts",
        "small_breasts",
        "flat_chest",
        "breasts"
      ],
      "framing": [
        "full_body",
        "cowboy_shot",
        "upper_body",
        "portrait",
        "close-up"
      ]
    },
    "color_parent_dedup_garments": [
      "hair",
      "eyes",
      "dress",
      "china_dress",
      "shirt",
      "jacket",
      "skirt",
      "shorts",
      "pants",
      "thighhighs",
      "kneehighs",
      "socks",
      "gloves",
      "boots",
      "shoes",
      "bikini",
      "bra",
      "panties",
      "bow",
      "ribbon",
      "tail",
      "ears"
    ]
  },
  "2_view_and_framing_conflicts": {
    "view_conflicts": {
      "front_view": {
        "incompatible_tags": [
          "from_behind",
          "back",
          "bare_back",
          "shoulder_blades",
          "nape",
          "spine",
          "back_dimples",
          "sacral_dimples",
          "butt",
          "ass",
          "ass_focus",
          "butt_crack",
          "backless_dress",
          "backless_outfit",
          "back_bow",
          "tailplug",
          "buttplug",
          "back_tattoo",
          "reverse_bunnysuit",
          "over_the_shoulder",
          "looking_back",
          "profile",
          "from_side",
          "side_view",
          "sideboob"
        ],
        "reason": "Forces rear or side rotation to render back/side anatomy."
      },
      "back_view": {
        "incompatible_tags": [
          "front_view",
          "straight-on",
          "navel",
          "stomach",
          "midriff",
          "abs",
          "cleavage",
          "underboob",
          "cameltoe",
          "bulge",
          "collarbone",
          "chest",
          "breasts",
          "large_breasts",
          "medium_breasts",
          "small_breasts",
          "flat_chest"
        ],
        "reason": "Forces torso to rotate forward to render front anatomy."
      },
      "eye_level": {
        "incompatible_tags": [
          "from_below",
          "upskirt",
          "pantyshot",
          "underboob",
          "undershot",
          "soles",
          "stepping_on_viewer",
          "from_above",
          "downblouse",
          "top_of_head"
        ],
        "reason": "Hijacks camera pitch (Z-axis) to extreme high or low angles."
      }
    },
    "framing_conflicts": {
      "portrait": {
        "incompatible_tags": [
          "full_body",
          "cowboy_shot",
          "lower_body",
          "feet",
          "barefoot",
          "toes",
          "shoes",
          "boots",
          "high_heels",
          "sneakers",
          "sandals",
          "footwear",
          "ankles",
          "anklet",
          "socks",
          "kneehighs",
          "thighhighs",
          "pantyhose",
          "legwear",
          "thighs",
          "thick_thighs",
          "thigh_gap",
          "hips",
          "wide_hips",
          "butt",
          "ass",
          "panties",
          "skirt",
          "shorts",
          "pants",
          "navel",
          "midriff",
          "standing",
          "walking",
          "running",
          "sitting",
          "squatting"
        ],
        "reason": "Pulls camera down/back to render lower body or full-body actions."
      },
      "upper_body": {
        "incompatible_tags": [
          "full_body",
          "lower_body",
          "feet",
          "barefoot",
          "toes",
          "shoes",
          "boots",
          "high_heels",
          "sneakers",
          "sandals",
          "footwear",
          "ankles",
          "anklet",
          "socks",
          "kneehighs",
          "thighhighs",
          "thighs",
          "thick_thighs",
          "knees",
          "calves",
          "shins",
          "standing",
          "walking",
          "running",
          "squatting"
        ],
        "reason": "Forces framing below the waist/hips."
      },
      "full_body": {
        "incompatible_tags": [
          "portrait",
          "close-up",
          "face_focus",
          "eye_focus",
          "macro_shot",
          "extreme_close-up"
        ],
        "reason": "Forces tight facial framing conflicting with full figure capture."
      }
    },
    "pose_conflicts": {
      "arms_down_or_hips": {
        "target_tags": [
          "arms_at_sides",
          "hands_on_hips",
          "hands_on_own_hips",
          "hands_in_pockets",
          "crossed_arms"
        ],
        "incompatible_tags": [
          "armpits",
          "bare_armpits",
          "armpit_crease",
          "armpit_hair",
          "arms_up",
          "hands_up",
          "stretching",
          "tying_hair",
          "arms_behind_head",
          "hands_behind_head",
          "reaching_towards_viewer",
          "peace_sign",
          "waving",
          "salute"
        ],
        "reason": "Armpits and gesture tags force arms to lift or break resting position."
      },
      "standing_upright": {
        "target_tags": [
          "standing",
          "straight_posture"
        ],
        "incompatible_tags": [
          "sitting",
          "seiza",
          "wariza",
          "indian_style",
          "hugging_knees",
          "lying",
          "on_back",
          "on_stomach",
          "on_side",
          "all_fours",
          "top-down_bottom-up",
          "bent_over",
          "leaning_forward",
          "squatting",
          "crouching",
          "kneeling",
          "soles"
        ],
        "reason": "Forces grounded, seated, or bent spine postures."
      }
    }
  },
  "3_breast_size_rules_and_recommendations": {
    "all_size_tiers": [
      "flat_chest",
      "small_breasts",
      "medium_breasts",
      "large_breasts",
      "huge_breasts",
      "gigantic_breasts",
      "hyper_breasts"
    ],
    "tiers": {
      "flat_chest": {
        "synonyms": [
          "aa_cup",
          "flat"
        ],
        "core_reinforcers": [
          "aa_cup",
          "small_underbust",
          "slender_torso"
        ],
        "clothing_interaction": [
          "loose_shirt",
          "baggy_clothes",
          "sarashi",
          "bandeau",
          "open_shirt",
          "oversized_shirt"
        ],
        "pose_and_anatomy": [
          "collarbone",
          "visible_ribs",
          "pectorals",
          "arms_crossed_over_chest"
        ],
        "strip_size_tags": [
          "breasts",
          "small_breasts",
          "medium_breasts",
          "large_breasts",
          "huge_breasts",
          "gigantic_breasts",
          "hyper_breasts",
          "a_cup",
          "b_cup",
          "c_cup",
          "d_cup",
          "e_cup",
          "f_cup",
          "g_cup",
          "h_cup",
          "voluptuous"
        ],
        "strip_hijackers": [
          "cleavage",
          "deep_cleavage",
          "underboob",
          "sideboob",
          "bouncing_breasts",
          "jiggling",
          "breast_press",
          "boob_shelf",
          "breast_rest",
          "taut_shirt",
          "button_gap",
          "popping_button",
          "paizuri",
          "hand_bra",
          "underbust",
          "framed_breasts",
          "strap_between_breasts",
          "seatbelt_between_breasts",
          "sagging_breasts",
          "hanging_breasts",
          "breast_curtains",
          "bursting_breasts",
          "boob_pocket",
          "virgin_killer_sweater"
        ],
        "auto_negative_lock": [
          "breasts",
          "cleavage",
          "medium_breasts",
          "large_breasts",
          "underboob",
          "sideboob",
          "boob_shadow"
        ]
      },
      "small_breasts": {
        "synonyms": [
          "a_cup",
          "b_cup",
          "petite_breasts",
          "perky_breasts"
        ],
        "core_reinforcers": [
          "a_cup",
          "b_cup",
          "perky_breasts",
          "petite_breasts"
        ],
        "clothing_interaction": [
          "tight_shirt",
          "crop_top",
          "sports_bra",
          "micro_bikini",
          "slight_cleavage",
          "unbuttoned_shirt"
        ],
        "pose_and_anatomy": [
          "collarbone",
          "slender",
          "arched_back"
        ],
        "strip_size_tags": [
          "flat_chest",
          "aa_cup",
          "medium_breasts",
          "large_breasts",
          "huge_breasts",
          "gigantic_breasts",
          "hyper_breasts",
          "c_cup",
          "d_cup",
          "e_cup",
          "f_cup",
          "g_cup",
          "h_cup",
          "voluptuous"
        ],
        "strip_hijackers": [
          "deep_cleavage",
          "underboob",
          "boob_shelf",
          "breast_rest",
          "breasts_on_table",
          "button_gap",
          "popping_button",
          "paizuri",
          "sagging_breasts",
          "hanging_breasts",
          "heavy_breasts",
          "bursting_breasts",
          "framed_breasts",
          "boob_pocket",
          "breasts_bigger_than_head"
        ],
        "auto_negative_lock": [
          "flat_chest",
          "large_breasts",
          "huge_breasts",
          "deep_cleavage",
          "sagging_breasts",
          "boob_shelf"
        ]
      },
      "medium_breasts": {
        "synonyms": [
          "c_cup",
          "d_cup",
          "average_breasts"
        ],
        "core_reinforcers": [
          "c_cup",
          "d_cup",
          "proportionate_body",
          "full_breasts"
        ],
        "clothing_interaction": [
          "cleavage",
          "fitted_shirt",
          "sweater",
          "bikini",
          "cleavage_cutout"
        ],
        "pose_and_anatomy": [
          "hourglass_figure",
          "perky_breasts",
          "bouncing_breasts"
        ],
        "strip_size_tags": [
          "flat_chest",
          "aa_cup",
          "small_breasts",
          "a_cup",
          "b_cup",
          "large_breasts",
          "huge_breasts",
          "gigantic_breasts",
          "hyper_breasts",
          "e_cup",
          "f_cup",
          "g_cup",
          "h_cup"
        ],
        "strip_hijackers": [
          "boob_shelf",
          "breast_rest",
          "breasts_on_table",
          "popping_button",
          "bursting_breasts",
          "sagging_breasts",
          "heavy_breasts",
          "breasts_bigger_than_head",
          "supporting_own_breasts",
          "visible_ribs",
          "pectorals"
        ],
        "auto_negative_lock": [
          "flat_chest",
          "small_breasts",
          "huge_breasts",
          "gigantic_breasts",
          "hyper_breasts"
        ]
      },
      "large_breasts": {
        "synonyms": [
          "e_cup",
          "f_cup",
          "full_breasts"
        ],
        "core_reinforcers": [
          "e_cup",
          "f_cup",
          "curvy",
          "voluptuous",
          "deep_cleavage"
        ],
        "clothing_interaction": [
          "taut_shirt",
          "taut_clothes",
          "button_gap",
          "cleavage_cutout",
          "underboob",
          "virgin_killer_sweater",
          "keyhole_turtleneck",
          "strap_between_breasts",
          "seatbelt_between_breasts",
          "boob_pocket"
        ],
        "pose_and_physics": [
          "bouncing_breasts",
          "jiggling",
          "arms_crossed_under_breasts",
          "framed_breasts",
          "breast_press",
          "leaning_forward"
        ],
        "strip_size_tags": [
          "flat_chest",
          "aa_cup",
          "small_breasts",
          "a_cup",
          "b_cup",
          "medium_breasts",
          "c_cup",
          "d_cup",
          "huge_breasts",
          "gigantic_breasts",
          "hyper_breasts"
        ],
        "strip_hijackers": [
          "visible_ribs",
          "pectorals",
          "flat",
          "boyish",
          "androgynous",
          "breasts_bigger_than_head",
          "breasts_bigger_than_torso",
          "hyper"
        ],
        "auto_negative_lock": [
          "flat_chest",
          "small_breasts",
          "medium_breasts",
          "petite"
        ]
      },
      "huge_breasts": {
        "synonyms": [
          "g_cup",
          "h_cup",
          "i_cup",
          "heavy_breasts"
        ],
        "core_reinforcers": [
          "g_cup",
          "h_cup",
          "heavy_breasts",
          "voluptuous",
          "deep_cleavage"
        ],
        "clothing_interaction": [
          "boob_shelf",
          "taut_clothes",
          "button_gap",
          "popping_button",
          "impossible_shirt",
          "vacuum_shirt",
          "underboob_cutout",
          "overflowing_breasts",
          "cropped_sweater",
          "micro_bikini"
        ],
        "pose_and_physics": [
          "sagging_breasts",
          "hanging_breasts",
          "breast_rest",
          "breasts_on_table",
          "holding_own_breasts",
          "arms_under_breasts",
          "breast_curtains",
          "heavy_breathing"
        ],
        "strip_size_tags": [
          "flat_chest",
          "aa_cup",
          "small_breasts",
          "a_cup",
          "b_cup",
          "medium_breasts",
          "c_cup",
          "d_cup",
          "large_breasts",
          "e_cup",
          "f_cup",
          "gigantic_breasts",
          "hyper_breasts"
        ],
        "strip_hijackers": [
          "visible_ribs",
          "pectorals",
          "perky_breasts",
          "petite",
          "skinny",
          "slender",
          "boyish",
          "androgynous"
        ],
        "auto_negative_lock": [
          "flat_chest",
          "small_breasts",
          "medium_breasts",
          "perky_breasts",
          "slender",
          "skinny"
        ]
      },
      "gigantic_breasts": {
        "synonyms": [
          "hyper_breasts",
          "j_cup",
          "k_cup",
          "massive_breasts",
          "enormous_breasts"
        ],
        "core_reinforcers": [
          "hyper_breasts",
          "massive_breasts",
          "enormous_breasts",
          "breasts_bigger_than_head",
          "voluptuous"
        ],
        "clothing_interaction": [
          "bursting_breasts",
          "tearing_clothes",
          "popping_button",
          "button_strain",
          "impossible_clothes",
          "boob_shelf",
          "underboob",
          "unbuttoned_shirt",
          "open_clothes",
          "string_bikini"
        ],
        "pose_and_physics": [
          "supporting_own_breasts",
          "two-handed_breast_hold",
          "breasts_on_table",
          "breast_rest",
          "sagging_breasts",
          "heavy_breasts",
          "veiny_breasts",
          "shiny_skin",
          "back_pain",
          "leaning_forward"
        ],
        "strip_size_tags": [
          "flat_chest",
          "aa_cup",
          "small_breasts",
          "medium_breasts",
          "large_breasts",
          "huge_breasts"
        ],
        "strip_hijackers": [
          "perky_breasts",
          "proportionate",
          "petite",
          "skinny",
          "slender"
        ],
        "auto_negative_lock": [
          "flat_chest",
          "small_breasts",
          "medium_breasts",
          "large_breasts",
          "proportionate",
          "petite",
          "skinny"
        ]
      },
      "hyper_breasts": {
        "synonyms": [
          "extreme_breasts",
          "breasts_bigger_than_torso",
          "impossible_breast_size"
        ],
        "core_reinforcers": [
          "hyper_breasts",
          "massive_breasts",
          "enormous_breasts",
          "breasts_bigger_than_head",
          "breasts_bigger_than_torso",
          "voluptuous"
        ],
        "clothing_interaction": [
          "bursting_breasts",
          "tearing_clothes",
          "impossible_clothes",
          "overflowing_breasts",
          "micro_bikini",
          "open_clothes"
        ],
        "pose_and_physics": [
          "supporting_own_breasts",
          "two-handed_breast_hold",
          "breasts_on_table",
          "breast_rest",
          "sagging_breasts",
          "heavy_breasts",
          "back_pain",
          "leaning_forward"
        ],
        "strip_size_tags": [
          "flat_chest",
          "aa_cup",
          "small_breasts",
          "medium_breasts",
          "large_breasts",
          "huge_breasts",
          "gigantic_breasts"
        ],
        "strip_hijackers": [
          "perky_breasts",
          "proportionate",
          "petite",
          "skinny",
          "slender"
        ],
        "auto_negative_lock": [
          "flat_chest",
          "small_breasts",
          "medium_breasts",
          "large_breasts",
          "huge_breasts",
          "proportionate",
          "petite",
          "skinny"
        ]
      }
    }
  },
  "4_body_and_limb_proportions": {
    "wide_hips_and_pelvis": {
      "target_tags": [
        "wide_hips",
        "broad_hips",
        "childbearing_hips",
        "pear_shaped",
        "huge_hips",
        "narrow_waist_wide_hips",
        "pronounced_hips",
        "wide_pelvis"
      ],
      "core_reinforcers": [
        "wide_hips",
        "curvy",
        "hourglass_figure",
        "narrow_waist_wide_hips",
        "pronounced_hips",
        "thick_thighs",
        "small_waist",
        "hip_bones"
      ],
      "clothing_interaction": [
        "highleg_panties",
        "highleg_bikini",
        "highleg_leotard",
        "highleg_swimsuit",
        "pelvic_curtain",
        "hip_vent",
        "side_slit",
        "high_side_slit",
        "low-rise_shorts",
        "low-rise_jeans",
        "low-rise_pants",
        "low-rise_skirt",
        "side-tie_panties",
        "side-tie_bikini",
        "whale_tail",
        "hip_chain",
        "waist_chain",
        "belly_chain",
        "bodycon_dress",
        "pencil_skirt",
        "tight_shorts",
        "hotpants",
        "micro_shorts",
        "denim_shorts",
        "taut_skirt",
        "hip_cutout",
        "pelvic_cutout",
        "sarong"
      ],
      "accessories_and_gear": [
        "hip_holster",
        "thigh_holster",
        "hip_bag",
        "fanny_pack",
        "hip_pouch",
        "tassets",
        "faulds",
        "hip_armor"
      ],
      "pose_and_physics": [
        "hips_cocked",
        "hip_sway",
        "contrapposto",
        "weight_on_one_leg",
        "hands_on_hips",
        "hands_on_own_hips",
        "hand_on_hip",
        "hand_on_another's_hip",
        "hips_pushed_back",
        "arched_back",
        "wide_stance",
        "crossed_legs",
        "sitting",
        "bent_over"
      ],
      "incompatible_tags": [
        "narrow_hips",
        "boyish_hips",
        "flat_hips",
        "straight_torso",
        "tubular_waist",
        "tubular_body",
        "thin_legs",
        "slender_legs",
        "stick_legs",
        "flat_ass",
        "small_ass",
        "androgynous",
        "baggy_pants",
        "oversized_pants",
        "harem_pants",
        "poofy_dress",
        "bell_skirt",
        "hoop_skirt",
        "balloon_skirt",
        "puffy_shorts",
        "pumpkin_pants",
        "wide-leg_pants"
      ],
      "auto_negative_lock": [
        "narrow_hips",
        "boyish_hips",
        "flat_hips",
        "straight_torso",
        "tubular_body",
        "stick_legs",
        "flat_ass",
        "slender_hips",
        "androgynous_lower_body"
      ]
    },
    "thick_curvy_lower": {
      "target_tags": [
        "thick_thighs",
        "wide_hips",
        "curvy",
        "plump_thighs",
        "bottom_heavy",
        "huge_ass",
        "pear_shaped"
      ],
      "clothing_interaction": [
        "skindentation",
        "thighhighs",
        "garter_straps",
        "thigh_strap",
        "tight_shorts",
        "denim_shorts",
        "pelvic_curtain",
        "hip_vent",
        "side_slit",
        "taut_skirt",
        "bike_shorts"
      ],
      "pose_and_physics": [
        "thigh_squish",
        "sitting",
        "crossed_legs",
        "hands_on_hips",
        "wide_stance",
        "hips_pushed_back"
      ],
      "incompatible_tags": [
        "thin_legs",
        "slender_legs",
        "stick_legs",
        "thigh_gap",
        "narrow_hips",
        "boyish_hips",
        "flat_ass",
        "small_ass",
        "skinny",
        "lanky",
        "frail"
      ],
      "auto_negative_lock": [
        "thin_legs",
        "stick_legs",
        "thigh_gap",
        "narrow_hips",
        "skinny",
        "slender_legs"
      ]
    },
    "slender_petite_lower": {
      "target_tags": [
        "slender",
        "skinny",
        "thin_legs",
        "thigh_gap",
        "narrow_hips",
        "petite"
      ],
      "incompatible_tags": [
        "thick_thighs",
        "wide_hips",
        "curvy",
        "voluptuous",
        "plump",
        "chubby",
        "huge_ass",
        "bottom_heavy",
        "skindentation",
        "thigh_squish",
        "muffin_top",
        "stomach_bulge",
        "belly"
      ],
      "auto_negative_lock": [
        "thick_thighs",
        "wide_hips",
        "chubby",
        "plump",
        "skindentation"
      ]
    },
    "muscular_toned": {
      "target_tags": [
        "abs",
        "toned",
        "muscular",
        "muscular_female",
        "six_pack",
        "athletic",
        "defined_muscles"
      ],
      "clothing_interaction": [
        "sports_bra",
        "crop_top",
        "midriff",
        "tank_top",
        "gym_shorts",
        "spats",
        "boxing_gloves",
        "hand_wraps"
      ],
      "pose_and_physics": [
        "flexing",
        "biceps",
        "obliques",
        "broad_shoulders",
        "sweat",
        "shiny_skin",
        "fighting_stance"
      ],
      "incompatible_tags": [
        "chubby",
        "plump",
        "soft_body",
        "belly",
        "stomach_bulge",
        "pudge",
        "muffin_top",
        "love_handles",
        "flabby",
        "frail",
        "skinny"
      ],
      "auto_negative_lock": [
        "chubby",
        "plump",
        "soft_body",
        "flabby",
        "frail",
        "skinny",
        "narrow_shoulders"
      ]
    },
    "soft_chubby": {
      "target_tags": [
        "chubby",
        "plump",
        "soft_body",
        "belly",
        "stomach_bulge",
        "muffin_top"
      ],
      "incompatible_tags": [
        "abs",
        "toned",
        "muscular",
        "six_pack",
        "obliques",
        "narrow_waist",
        "slim_waist",
        "flat_stomach",
        "skinny",
        "slender",
        "visible_ribs"
      ],
      "auto_negative_lock": [
        "abs",
        "muscular",
        "skinny",
        "slender",
        "visible_ribs"
      ]
    },
    "massive_forearms_and_hands": {
      "core_reinforcers": [
        "oversized_forearms",
        "massive_forearms",
        "thick_arms",
        "large_hands",
        "exaggerated_proportions"
      ],
      "recommended_clothing": [
        "short_sleeves",
        "sleeveless",
        "bare_arms",
        "wristbands",
        "gauntlets",
        "forearm_guards",
        "fingerless_gloves",
        "rolled_up_sleeves"
      ],
      "recommended_poses": [
        "clenched_fists",
        "fighting_stance",
        "arms_crossed",
        "hands_on_hips",
        "flexing",
        "punching"
      ],
      "strip_conflicts": [
        "slender_arms",
        "thin_arms",
        "delicate_hands",
        "dainty",
        "long_sleeves",
        "sleeves_past_wrists",
        "sleeves_past_fingers"
      ],
      "auto_negative_lock": [
        "thin_arms",
        "skinny_arms",
        "small_hands",
        "delicate_limbs",
        "realistic_proportions"
      ]
    },
    "thick_calves_and_legs": {
      "core_reinforcers": [
        "thick_calves",
        "thick_thighs",
        "bottom_heavy",
        "sturdy_legs"
      ],
      "recommended_clothing": [
        "loose_socks",
        "leg_warmers",
        "chunky_boots",
        "sneakers",
        "thighhighs",
        "skindentation"
      ],
      "strip_conflicts": [
        "thin_legs",
        "stick_legs",
        "slender_legs",
        "thigh_gap",
        "stilettos"
      ],
      "auto_negative_lock": [
        "thin_legs",
        "skinny_legs",
        "thigh_gap"
      ]
    }
  },
  "5_stature_and_species_rules": {
    "stature_conflicts": {
      "short_petite": {
        "target_tags": [
          "short",
          "petite",
          "loli",
          "small_stature",
          "chibi"
        ],
        "incompatible_tags": [
          "tall",
          "tall_female",
          "long_legs",
          "model_body",
          "mature_female",
          "milf",
          "statuesque",
          "towering",
          "high_heels",
          "stilettos"
        ],
        "framing_hijackers": [
          "huge_weapon",
          "giant_weapon",
          "polearm",
          "staff",
          "oversized_weapon",
          "scythe"
        ]
      },
      "tall_statuesque": {
        "target_tags": [
          "tall",
          "tall_female",
          "long_legs",
          "mature_female",
          "statuesque"
        ],
        "incompatible_tags": [
          "short",
          "petite",
          "loli",
          "chibi",
          "big_head",
          "round_face",
          "oversized_shirt",
          "oversized_clothes",
          "sleeves_past_fingers",
          "sleeves_past_wrists"
        ]
      }
    },
    "kemonomimi_human_skin": {
      "target_tags": [
        "kemonomimi_mode",
        "animal_ears",
        "rabbit_ears",
        "fox_ears",
        "cat_ears",
        "dog_ears",
        "tanuki_ears",
        "wolf_ears",
        "horse_ears"
      ],
      "strip_furry_bleed": [
        "furry",
        "furry_female",
        "furry_male",
        "anthro",
        "fur",
        "white_fur",
        "pink_fur",
        "black_fur",
        "brown_fur",
        "grey_fur",
        "orange_fur",
        "blue_fur",
        "snout",
        "muzzle",
        "animal_nose",
        "paws",
        "paw_pads",
        "digitigrade",
        "furry_body",
        "body_fur"
      ],
      "strip_four_ears_triggers": [
        "human_ears",
        "earrings",
        "hoop_earrings",
        "stud_earrings",
        "ear_piercing",
        "hair_tucked_behind_ear",
        "elf_ears",
        "pointy_ears"
      ]
    },
    "kemonomimi_mood_synergy": {
      "happy_playful": {
        "detected_expressions": [
          "smile",
          "grin",
          "3",
          "cat_mouth",
          "laughing",
          "smug",
          "wink"
        ],
        "recommend_ear_tail_tags": [
          "perked_ears",
          "animal_ear_fluff",
          "wagging_tail",
          "tail_raised",
          "fluffy_tail"
        ],
        "strip_conflicts": [
          "droopy_ears",
          "ears_back",
          "tail_between_legs"
        ]
      },
      "shy_embarrassed": {
        "detected_expressions": [
          "blush",
          "embarrassed",
          "nervous",
          "pout",
          "tears",
          "looking_down"
        ],
        "recommend_ear_tail_tags": [
          "droopy_ears",
          "folded_ears",
          "ears_down",
          "tail_between_legs",
          "hugging_own_tail"
        ],
        "strip_conflicts": [
          "perked_ears",
          "wagging_tail"
        ]
      },
      "front_view_tail_visibility": {
        "detected_view": [
          "front_view",
          "straight-on",
          "facing_viewer"
        ],
        "recommend_ear_tail_tags": [
          "tail_to_side",
          "tail_around_leg",
          "visible_tail",
          "holding_own_tail"
        ]
      }
    }
  },
  "6_facial_state_and_markings": {
    "closed_eyes": {
      "target_tags": [
        "closed_eyes",
        "eyes_closed",
        "sleeping"
      ],
      "incompatible_tags": [
        "looking_at_viewer",
        "looking_away",
        "looking_back",
        "looking_up",
        "looking_down",
        "eye_contact",
        "staring",
        "glaring",
        "half-closed_eyes",
        "jitome",
        "tareme",
        "tsurime"
      ]
    },
    "closed_mouth": {
      "target_tags": [
        "closed_mouth",
        "3",
        "cat_mouth",
        "pout",
        "tight_lips"
      ],
      "incompatible_tags": [
        "open_mouth",
        "parted_lips",
        "slightly_open_mouth",
        "teeth",
        "visible_teeth",
        "clenched_teeth",
        "shark_teeth",
        "tongue",
        "tongue_out",
        "uvula",
        "saliva",
        "drooling",
        "laughing",
        "shouting",
        "yawning",
        "sigh"
      ]
    },
    "single_eyewear_rule": {
      "target_tags": [
        "glasses",
        "sunglasses",
        "round_eyewear",
        "rimmed_eyewear",
        "semi-rimless_eyewear",
        "under-rim_eyewear",
        "pince-nez"
      ],
      "incompatible_tags": [
        "eyewear_on_head",
        "glasses_on_head",
        "sunglasses_on_head",
        "goggles_on_head",
        "holding_eyewear",
        "removing_eyewear",
        "eyewear_hang"
      ],
      "reason": "Prevents duplicating eyewear simultaneously on the face and resting on top of the head or held in hands."
    },
    "clean_scar_lock": {
      "target_tags": [
        "scar",
        "scar_on_face",
        "scar_across_eye",
        "scar_on_cheek",
        "scar_on_nose"
      ],
      "strip_gore_escalation": [
        "blood",
        "bleeding",
        "blood_on_face",
        "open_wound",
        "wound",
        "guro",
        "injured",
        "bruise",
        "bandages",
        "eyepatch"
      ],
      "auto_negative_lock": [
        "blood",
        "bleeding",
        "gore",
        "fresh_wound",
        "bandages"
      ]
    }
  },
  "7_style_skin_and_color_protection": {
    "surface_and_style_tiers": {
      "flat_2d_anime": {
        "core_reinforcers": [
          "anime_style",
          "cel_shading",
          "flat_color",
          "bold_outlines",
          "2d"
        ],
        "recommended_details": [
          "blush_stickers",
          "hash_blush",
          "high_contrast",
          "clean_lineart"
        ],
        "strip_conflicts": [
          "3d",
          "realistic",
          "photorealistic",
          "subsurface_scattering",
          "shiny_skin",
          "oily_skin",
          "pores",
          "lips",
          "depth_of_field"
        ],
        "auto_negative_lock": [
          "3d",
          "realistic",
          "photorealistic",
          "shiny_skin",
          "airbrush",
          "uncanny"
        ]
      },
      "glossy_2_5d_illustration": {
        "core_reinforcers": [
          "illustration",
          "soft_shading",
          "detailed_shading",
          "volumetric_lighting"
        ],
        "recommended_details": [
          "shiny_skin",
          "glossy_skin",
          "rim_lighting",
          "backlighting",
          "specular_highlights",
          "depth_of_field",
          "bloom"
        ],
        "strip_conflicts": [
          "flat_color",
          "cel_shading",
          "sketch",
          "monochrome",
          "lineart"
        ],
        "auto_negative_lock": [
          "flat_color",
          "sketch",
          "unfinished",
          "photorealistic",
          "real_life"
        ]
      }
    },
    "dark_skin_protection": {
      "target_tags": [
        "dark_skin",
        "dark-skinned_female",
        "dark-skinned_male",
        "tan",
        "brown_skin"
      ],
      "incompatible_tags": [
        "pale_skin",
        "fair_skin",
        "white_skin",
        "porcelain_skin",
        "albino",
        "overexposed",
        "bleached"
      ],
      "caution_tags": [
        "tanlines",
        "sunburn",
        "full-face_blush"
      ]
    },
    "color_bleed_protection": {
      "uncolored_garment_watch": [
        "dress",
        "shirt",
        "blouse",
        "jacket",
        "coat",
        "hoodie",
        "sweater",
        "crop_top",
        "skirt",
        "shorts",
        "pants",
        "thighhighs",
        "boots",
        "shoes",
        "gloves"
      ],
      "multi_color_garment_modifiers": [
        "two-tone_clothes",
        "colored_inner_lining",
        "blue_lining",
        "red_lining",
        "purple_trim",
        "gold_trim",
        "white_trim",
        "black_trim",
        "gradient_clothes"
      ],
      "white_on_white_silhouette_fix": {
        "when_tags_any": [
          "white_hair",
          "white_dress",
          "white_shirt",
          "white_boots",
          "white_fur"
        ],
        "and_background": [
          "white_background"
        ],
        "recommend_fixes": [
          "grey_background",
          "simple_background",
          "bold_outlines",
          "drop_shadow"
        ]
      }
    }
  },
  "8_combo_synergy_matrix": [
    {
      "when_anatomy_any": [
        "large_breasts",
        "huge_breasts",
        "gigantic_breasts",
        "hyper_breasts"
      ],
      "and_clothing_any": [
        "shirt",
        "blouse",
        "collared_shirt",
        "dress_shirt",
        "cardigan",
        "vest"
      ],
      "recommend_tags": [
        "button_gap",
        "popping_button",
        "taut_shirt",
        "button_strain"
      ]
    },
    {
      "when_anatomy_any": [
        "large_breasts",
        "huge_breasts",
        "gigantic_breasts"
      ],
      "and_clothing_any": [
        "sweater",
        "turtleneck",
        "ribbed_sweater",
        "virgin_killer_sweater"
      ],
      "recommend_tags": [
        "boob_shelf",
        "taut_clothes",
        "framed_breasts",
        "cleavage_cutout"
      ]
    },
    {
      "when_anatomy_any": [
        "thick_thighs",
        "wide_hips",
        "plump_thighs"
      ],
      "and_clothing_any": [
        "thighhighs",
        "kneehighs",
        "pantyhose",
        "garter_straps",
        "thigh_strap"
      ],
      "recommend_tags": [
        "skindentation",
        "thigh_squish",
        "taut_legwear"
      ]
    },
    {
      "when_anatomy_any": [
        "thick_thighs",
        "wide_hips",
        "huge_ass"
      ],
      "and_clothing_any": [
        "shorts",
        "denim_shorts",
        "hotpants",
        "micro_shorts"
      ],
      "recommend_tags": [
        "tight_shorts",
        "muffin_top",
        "low-rise_shorts",
        "frayed_shorts",
        "booty_shorts"
      ]
    },
    {
      "when_anatomy_any": [
        "wide_hips",
        "broad_hips",
        "pear_shaped",
        "huge_hips",
        "narrow_waist_wide_hips"
      ],
      "and_clothing_any": [
        "china_dress",
        "qipao",
        "cheongsam",
        "dress",
        "skirt"
      ],
      "recommend_tags": [
        "hip_vent",
        "high_side_slit",
        "side_slit",
        "pelvic_curtain",
        "tight_dress",
        "bodycon_dress"
      ]
    },
    {
      "when_anatomy_any": [
        "wide_hips",
        "broad_hips",
        "hourglass_figure"
      ],
      "and_clothing_any": [
        "bikini",
        "swimsuit",
        "leotard",
        "bodysuit",
        "panties",
        "underwear"
      ],
      "recommend_tags": [
        "highleg_bikini",
        "highleg_panties",
        "highleg_leotard",
        "side-tie_bikini",
        "hip_chain"
      ]
    },
    {
      "when_anatomy_any": [
        "wide_hips",
        "narrow_waist_wide_hips"
      ],
      "and_clothing_any": [
        "pants",
        "jeans",
        "shorts",
        "denim_shorts"
      ],
      "recommend_tags": [
        "tight_shorts",
        "muffin_top",
        "low-rise_shorts",
        "frayed_shorts",
        "booty_shorts"
      ]
    },
    {
      "when_anatomy_any": [
        "wide_hips",
        "pronounced_hips",
        "broad_hips",
        "narrow_waist_wide_hips"
      ],
      "and_pose_any": [
        "standing",
        "front_view",
        "full_body",
        "cowboy_shot"
      ],
      "recommend_tags": [
        "hips_cocked",
        "hand_on_hip",
        "hands_on_hips",
        "contrapposto",
        "hip_sway"
      ]
    },
    {
      "when_anatomy_any": [
        "wide_hips",
        "narrow_waist_wide_hips"
      ],
      "and_accessory_any": [
        "chain",
        "belt",
        "holster",
        "bag"
      ],
      "recommend_tags": [
        "hip_chain",
        "waist_chain",
        "belly_chain",
        "hip_holster",
        "hip_bag"
      ]
    },
    {
      "when_anatomy_any": [
        "short",
        "petite",
        "loli"
      ],
      "and_prop_any": [
        "staff",
        "polearm",
        "huge_weapon",
        "giant_weapon",
        "scythe",
        "greatsword"
      ],
      "recommend_tags": [
        "upper_body",
        "cowboy_shot",
        "cropped_weapon",
        "head_in_frame"
      ],
      "auto_negative": [
        "wide_shot",
        "distant_shot",
        "tiny_character"
      ]
    }
  ],
  "9_canvas_grounding_and_presets": {
    "canvas_isolation_modes": {
      "single_image_lock": {
        "auto_positive": [
          "solo",
          "single_view"
        ],
        "auto_negative": [
          "multiple_views",
          "reference_sheet",
          "character_sheet",
          "turnaround",
          "split_screen",
          "comic",
          "4koma",
          "inset",
          "cut-in",
          "zoom_layer",
          "2girls",
          "multiple_girls",
          "border",
          "frame",
          "speech_bubble"
        ],
        "strip_from_positive": [
          "multiple_views",
          "reference_sheet",
          "character_sheet",
          "turnaround",
          "comic",
          "inset"
        ]
      },
      "reference_sheet_lock": {
        "auto_positive": [
          "reference_sheet",
          "character_sheet",
          "multiple_views",
          "turnaround",
          "full_body",
          "simple_background",
          "white_background"
        ],
        "auto_negative": [
          "complex_background",
          "scenery",
          "dynamic_angle",
          "foreshortening",
          "cropped",
          "close-up"
        ]
      },
      "png_tuber_cutout": {
        "positive_lock": [
          "solo",
          "single_view",
          "centered",
          "simple_background",
          "clean_outlines",
          "even_lighting",
          "looking_at_viewer"
        ],
        "negative_lock": [
          "depth_of_field",
          "blurry",
          "vignette",
          "particles",
          "lens_flare",
          "cropped_head",
          "cropped_arms",
          "drop_shadow",
          "complex_background"
        ]
      }
    },
    "grounding_and_floor_fixes": {
      "full_body_surface_lock": {
        "trigger_when_all": [
          "full_body"
        ],
        "recommend_grounding": [
          "standing_on_ground",
          "ground_shadow",
          "cast_shadow",
          "feet_visible"
        ],
        "strip_sinking_triggers": [
          "wading",
          "in_water",
          "ankle-deep",
          "knee-deep",
          "submerged",
          "buried_in_sand",
          "tall_grass",
          "cropped_legs"
        ]
      }
    }
  },
  "10_token_ordering_buckets": {
    "bucket_1_quality_and_count": [
      "masterpiece",
      "best_quality",
      "1girl",
      "solo",
      "single_view"
    ],
    "bucket_2_camera_and_framing": [
      "front_view",
      "straight-on",
      "symmetrical",
      "full_body",
      "cowboy_shot",
      "upper_body",
      "portrait",
      "from_above",
      "from_below"
    ],
    "bucket_3_character_and_species": [
      "yukia",
      "rabbit_girl",
      "kemonomimi_mode",
      "animal_ears",
      "rabbit_ears",
      "tanuki_ears",
      "tail",
      "dark_skin"
    ],
    "bucket_4_core_proportions": [
      "hyper_breasts",
      "gigantic_breasts",
      "huge_breasts",
      "large_breasts",
      "medium_breasts",
      "small_breasts",
      "flat_chest",
      "thick_thighs",
      "wide_hips",
      "broad_hips",
      "narrow_waist_wide_hips",
      "curvy",
      "slender",
      "muscular_female",
      "oversized_forearms",
      "massive_forearms"
    ],
    "bucket_5_hair_and_face": [
      "white_hair",
      "pink_hair",
      "long_hair",
      "bangs",
      "red_eyes",
      "amber_eyes",
      "smile",
      "3",
      "blush",
      "looking_at_viewer",
      "glasses"
    ],
    "bucket_6_outfit_and_synergy": [
      "china_dress",
      "crop_top",
      "denim_shorts",
      "thighhighs",
      "boots",
      "taut_clothes",
      "button_gap",
      "skindentation",
      "boob_shelf"
    ],
    "bucket_7_pose_and_action": [
      "standing",
      "standing_on_ground",
      "hands_on_hips",
      "sitting",
      "arms_under_breasts",
      "thigh_squish"
    ],
    "bucket_8_background_and_lighting": [
      "simple_background",
      "white_background",
      "ground_shadow",
      "volumetric_lighting",
      "rim_lighting"
    ]
  }
}