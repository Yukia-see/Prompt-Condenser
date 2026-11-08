
// ---------------------------------------------------------------------------
// Tests (ports of test_prompt_condenser.py)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    const WD: &str = "1girl, solo, blue_dress, sitting, from_above, blonde_hair, cleaning, thick_thighs";
    const MD1: &str = "1. Scene & Lighting: a bright room with a couch\n2. Core Pose: she is standing, seen from below";
    const MD2: &str = "1. Hair: long red hair\n2. Clothing: a red dress and black boots";
    const OK_JSON: &str = r#"{"scene_prompt":"s","character_prompt":"c"}"#;

    fn vocab(words: &[&str]) -> HashSet<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    fn md_ok(prompt: &str, _t: f32) -> Result<String, String> {
        Ok(if prompt.contains("space") { MD1 } else { MD2 }.to_string())
    }

    fn opts(mode: Mode, llm: bool, vision: bool) -> PipelineOptions {
        PipelineOptions { mode, use_llm_synthesis: llm, gemma_vision_pass: vision }
    }

    fn reply(text: &str, reason: Option<&str>) -> Result<GemmaReply, String> {
        Ok(GemmaReply { text: text.to_string(), finish_reason: reason.map(String::from) })
    }

    #[test]
    fn router_whole_words_and_underscores() {
        // "bedhead" must NOT match the scene keyword "bed"; "bedroom" is its own keyword.
        let (p1, p2) = sort_tag_lists(
            "cleaning, depressed, sitting, thick_thighs, from_below, bedhead, bedroom",
            Mode::SceneShape,
        );
        assert_eq!(p1.join(", "), "sitting, thick thighs, from below, bedroom");
        assert!(p2.contains(&"cleaning".to_string()));
        assert!(p2.contains(&"depressed".to_string()));
        assert!(p2.contains(&"bedhead".to_string()));
    }

    #[test]
    fn mode_b_sends_body_shape_to_pass2() {
        let (p1, p2) = sort_tag_lists("sitting, thick_thighs, arms up", Mode::ScenePose);
        assert_eq!(p1.join(", "), "sitting, arms up");
        assert_eq!(p2.join(", "), "thick thighs");
    }

    #[test]
    fn unknown_mode_is_an_error_and_known_modes_round_trip() {
        assert!(Mode::parse("bad").unwrap_err().contains("Unknown dual_pass_mode 'bad'"));
        for m in Mode::ALL {
            assert_eq!(Mode::parse(m.as_str()), Ok(m));
        }
    }

    #[test]
    fn prose_to_tags_longest_match() {
        let mut v = vocab(&["red dress", "black boots", "boots"]);
        v.extend(BUILTIN_VOCAB.iter().map(|s| s.to_string()));
        let a = analyze_prose_with_vocab("wearing a red dress and black boots", &v);
        assert_eq!(a.tags, vec!["red dress", "black boots"]);
    }

    #[test]
    fn wd14_wins_colors_moondream_wins_pose() {
        let v = vocab(&["red hair", "red dress", "black boots", "boots"]);
        let m = merge_tags(WD, MD1, MD2, Mode::SceneShape, Some(&v));
        assert_eq!(m.dropped_vlm, vec!["red dress", "red hair"]);
        assert_eq!(m.dropped_moondream, vec!["red dress", "red hair"]);
        assert_eq!(m.dropped_wd14, vec!["from above", "sitting"]);
    }

    #[test]
    fn no_llm_output_is_pure_tags() {
        let v = vocab(&["red hair", "red dress", "black boots", "boots"]);
        let out = process_direct(WD, MD1, MD2, Mode::SceneShape, Some(&v), &Config::default());
        assert_eq!(out.scene_prompt, "masterpiece, best quality, couch, standing, from below, thick thighs");
        assert_eq!(out.character_prompt, "1girl, solo, black boots, blue dress, blonde hair, cleaning");
    }

    #[test]
    fn parser_handles_preamble_lists_and_bad_json() {
        let cfg = Config::default();
        let good = parse_gemma_json_response(
            "Here:\n```json\n{\"scene_prompt\":\"a\",\"character_prompt\":[\"b\",\"c\"]}\n```",
            &cfg,
        );
        assert_eq!(good, Prompts { scene_prompt: "a".into(), character_prompt: "b, c".into() });
        assert_eq!(parse_gemma_json_response("[1,2]", &cfg).scene_prompt, cfg.quality_prefix);
        assert_eq!(parse_gemma_json_response("not json", &cfg).character_prompt, "not json");
    }

    #[test]
    fn llm_path_retries_once_on_truncation_and_requests_image() {
        let calls: RefCell<Vec<(u32, bool)>> = RefCell::new(Vec::new());
        let gemma = |r: &GemmaRequest| {
            calls.borrow_mut().push((r.max_tokens, r.with_image));
            if calls.borrow().len() == 1 {
                reply("{\"scene_prompt\":\"s\",\"char", Some("length"))
            } else {
                reply(OK_JSON, Some("stop"))
            }
        };
        let out = run_dual_pass(WD, &opts(Mode::ScenePose, true, true), &Config::default(), None, &md_ok, Some(&gemma));
        assert_eq!((out.scene_prompt.as_str(), out.character_prompt.as_str()), ("s", "c"));
        assert!(out.warnings.is_empty());
        assert_eq!(*calls.borrow(), vec![(GEMMA_MAX_TOKENS, true), (GEMMA_MAX_TOKENS_RETRY, true)]);
    }

    #[test]
    fn no_image_requested_without_vision_pass() {
        let seen = Cell::new(true);
        let gemma = |r: &GemmaRequest| {
            seen.set(r.with_image);
            reply(OK_JSON, None)
        };
        run_dual_pass(WD, &opts(Mode::SceneShape, true, false), &Config::default(), None, &md_ok, Some(&gemma));
        assert!(!seen.get());
    }

    #[test]
    fn vision_instruction_only_added_on_vision_pass() {
        let with = build_payload(WD, MD1, MD2, Mode::SceneShape, None, true);
        let without = build_payload(WD, MD1, MD2, Mode::SceneShape, None, false);
        assert!(with.system_instruction.contains("final check"));
        assert!(!without.system_instruction.contains("final check"));
    }

    #[test]
    fn defaults() {
        assert_eq!(GEMMA_OLLAMA_MODEL, "huihui_ai/gemma-4-abliterated:12b");
        assert_eq!(GEMMA_CONTEXT_WINDOW, 131072);
        assert_eq!(GEMMA_MAX_TOKENS, 768);
        assert_eq!(GEMMA_MAX_TOKENS_RETRY, 1024);
    }

    #[test]
    fn moondream_empty_pass_retries_then_uses_wd14_only() {
        let temps: RefCell<Vec<f32>> = RefCell::new(Vec::new());
        let empty = |_p: &str, t: f32| {
            temps.borrow_mut().push(t);
            Ok(String::new())
        };
        let out = run_dual_pass(
            "1girl, solo, sitting, blue_dress",
            &opts(Mode::SceneShape, false, false),
            &Config::default(),
            None,
            &empty,
            None,
        );
        assert_eq!(temps.borrow().len(), 4); // 2 passes x (1 try + 1 retry)
        assert!((temps.borrow()[1] - (VLM_TEMPERATURE + VLM_RETRY_TEMP_BUMP)).abs() < 1e-6);
        assert_eq!(out.character_prompt, "1girl, solo, blue dress");
        assert!(out.scene_prompt.contains("sitting"));
        assert_eq!(out.warnings.len(), 2);
    }

    #[test]
    fn moondream_refusal_retries_and_recovers() {
        let n = Cell::new(0);
        let flaky = |_p: &str, _t: f32| {
            n.set(n.get() + 1);
            Ok(match n.get() {
                1 => "I'm sorry, I cannot help with that.".to_string(),
                2 => MD1.to_string(),
                _ => MD2.to_string(),
            })
        };
        let out = run_dual_pass(WD, &opts(Mode::SceneShape, false, false), &Config::default(), None, &flaky, None);
        assert!(out.scene_prompt.contains("standing"));
        assert!(out.warnings.is_empty());
    }

    #[test]
    fn attribute_groups_wd14_wins() {
        let m = merge_tags("short_hair, 1girl", "1. Hair: she has long hair and a smile", "", Mode::SceneShape, None);
        assert!(m.dropped_vlm.contains(&"long hair".to_string()));
        assert!(m.dropped_moondream.contains(&"long hair".to_string()));
        let all: Vec<&String> = m.pass1_tags.iter().chain(m.pass2_tags.iter()).collect();
        assert!(all.contains(&&"short hair".to_string()));
    }

    #[test]
    fn unusable_gemma_output_falls_back_to_mechanical_merge() {
        let expected = process_direct(WD, MD1, MD2, Mode::SceneShape, None, &Config::default());
        let gemma = |_r: &GemmaRequest| reply("total garbage", None);
        let out = run_dual_pass(WD, &opts(Mode::SceneShape, true, false), &Config::default(), None, &md_ok, Some(&gemma));
        assert_eq!(out.scene_prompt, expected.scene_prompt);
        assert_eq!(out.character_prompt, expected.character_prompt);
        assert!(!out.warnings.is_empty());
    }

    #[test]
    fn gemma_empty_character_prompt_falls_back() {
        let expected = process_direct(WD, MD1, MD2, Mode::ScenePose, None, &Config::default());
        let gemma = |_r: &GemmaRequest| reply(r#"{"scene_prompt":"x","character_prompt":""}"#, None);
        let out = run_dual_pass(WD, &opts(Mode::ScenePose, true, false), &Config::default(), None, &md_ok, Some(&gemma));
        assert_eq!(out.scene_prompt, expected.scene_prompt);
        assert_eq!(out.character_prompt, expected.character_prompt);
        assert!(!out.warnings.is_empty());
    }

    #[test]
    fn parenthesis_escaping_option() {
        let cfg = Config { escape_parentheses: true, ..Config::default() };
        let out = process_direct("kafuu chino (gochiusa), 1girl", MD1, MD2, Mode::SceneShape, None, &cfg);
        assert!(out.character_prompt.contains("kafuu chino \\(gochiusa\\)"));
        assert_eq!(out.character_prompt.matches("\\(").count(), 1); // not double-escaped
        assert_eq!(style_tags("a \\(b\\) (c)", true), "a \\(b\\) \\(c\\)");
    }

    #[test]
    fn scene_keywords_route_common_background_tags() {
        let (p1, p2) = sort_tag_lists(
            "simple_background, white_background, night, cowboy_shot, 1girl, blue_dress",
            Mode::SceneShape,
        );
        assert_eq!(p1.join(", "), "simple background, white background, night, cowboy shot");
        assert_eq!(p2.join(", "), "1girl, blue dress");
    }

    #[test]
    fn negation_drops_tag_and_scope_ends_at_but() {
        let v = vocab(&["glasses", "red hat", "hat", "boots"]);
        let a = analyze_prose_with_vocab("She has no glasses but a red hat, not wearing boots", &v);
        assert!(a.negated.contains(&"glasses".to_string()) && a.negated.contains(&"boots".to_string()));
        assert_eq!(a.tags, vec!["red hat"]);
    }

    #[test]
    fn bare_is_not_a_negator() {
        let v = vocab(&["bare shoulders"]);
        assert_eq!(analyze_prose_with_vocab("bare shoulders", &v).tags, vec!["bare shoulders"]);
    }

    #[test]
    fn hedged_tag_needs_wd14_confirmation() {
        let v = vocab(&["glasses", "smile"]);
        let md1 = "1. Face: she appears to be wearing glasses and has a smile";
        let kept = merge_tags("1girl, glasses", md1, "", Mode::SceneShape, Some(&v));
        let dropped = merge_tags("1girl", md1, "", Mode::SceneShape, Some(&v));
        let all = |m: &Merged| m.pass1_tags.iter().chain(m.pass2_tags.iter()).cloned().collect::<Vec<_>>();
        assert!(all(&kept).contains(&"glasses".to_string()));
        assert!(!all(&dropped).contains(&"glasses".to_string()));
        assert!(dropped.dropped_hedged.contains(&"glasses".to_string()));
        assert!(all(&dropped).contains(&"smile".to_string()));
    }

    #[test]
    fn hedge_and_negation_scope_ends_at_new_statement() {
        let v = vocab(&["glasses", "hat", "smile"]);
        let a = analyze_prose_with_vocab("she appears to be wearing glasses and a hat", &v);
        assert_eq!(a.weak, ["glasses", "hat"].iter().map(|s| s.to_string()).collect());
        assert!(a.strong.is_empty());
        let b = analyze_prose_with_vocab("she appears to be wearing glasses and has a smile", &v);
        assert_eq!(b.weak, ["glasses"].iter().map(|s| s.to_string()).collect());
        assert_eq!(b.strong, ["smile"].iter().map(|s| s.to_string()).collect());
        let c = analyze_prose_with_vocab("she has no glasses and has a hat", &v);
        assert_eq!(c.negated, vec!["glasses"]);
        assert_eq!(c.tags, vec!["hat"]);
    }

    #[test]
    fn gemma_error_falls_back_with_warning() {
        let expected = process_direct(WD, MD1, MD2, Mode::SceneShape, None, &Config::default());
        let boom = |_r: &GemmaRequest| Err("connection refused".to_string());
        let out = run_dual_pass(WD, &opts(Mode::SceneShape, true, false), &Config::default(), None, &md_ok, Some(&boom));
        assert_eq!(out.scene_prompt, expected.scene_prompt);
        assert_eq!(out.character_prompt, expected.character_prompt);
        assert!(out.warnings.iter().any(|w| w.contains("connection refused")));
    }

    #[test]
    fn moondream_error_becomes_wd14_only_pass_with_warning() {
        let boom = |_p: &str, _t: f32| Err("model crashed".to_string());
        let out = run_dual_pass(
            "1girl, solo, sitting, blue_dress",
            &opts(Mode::SceneShape, false, false),
            &Config::default(),
            None,
            &boom,
            None,
        );
        assert_eq!(out.character_prompt, "1girl, solo, blue dress");
        assert_eq!(out.warnings.iter().filter(|w| w.contains("model crashed")).count(), 2);
    }

    #[test]
    fn missing_gemma_backend_uses_mechanical_merge_with_warning() {
        let expected = process_direct(WD, MD1, MD2, Mode::SceneShape, None, &Config::default());
        let out = run_dual_pass(WD, &opts(Mode::SceneShape, true, false), &Config::default(), None, &md_ok, None);
        assert_eq!(out.scene_prompt, expected.scene_prompt);
        assert!(out.warnings.iter().any(|w| w.contains("No Gemma backend")));
    }

    #[test]
    fn vision_toggle_without_llm_is_flagged() {
        let out = run_dual_pass(WD, &opts(Mode::SceneShape, false, true), &Config::default(), None, &md_ok, None);
        assert!(out.warnings.iter().any(|w| w.contains("gemma_vision_pass ignored")));
    }

    #[test]
    fn ollama_helpers() {
        let tags = r#"{"models":[{"name":"other:7b"},{"name":"huihui_ai/gemma-4-abliterated:12b"}]}"#;
        assert_eq!(ollama_model_pulled(tags, GEMMA_OLLAMA_MODEL), Ok(true));
        assert_eq!(ollama_model_pulled(tags, "huihui_ai/gemma-4-abliterated:26b"), Ok(false));
        assert_eq!(ollama_model_pulled(r#"{"models":[{"name":"x:latest"}]}"#, "x"), Ok(true));
        assert!(ollama_model_pulled("nope", "x").is_err());
        assert!(ollama_not_pulled_message("m:1").contains("ollama pull m:1"));

        let body = ollama_chat_request_body("m", "hi \"there\"\n", 0.35, 131072, 768, Some("QUJD"), true);
        let v = json::parse(&body).expect("request body must be valid JSON");
        assert_eq!(v.get("stream"), Some(&json::Value::Bool(false)));
        assert_eq!(v.get("format"), Some(&json::Value::Str("json".into())));
        let opts_v = v.get("options").unwrap();
        assert_eq!(opts_v.get("num_ctx"), Some(&json::Value::Num(131072.0)));
        assert_eq!(opts_v.get("num_predict"), Some(&json::Value::Num(768.0)));
        match v.get("messages") {
            Some(json::Value::Arr(m)) => {
                assert_eq!(m[0].get("content"), Some(&json::Value::Str("hi \"there\"\n".into())));
                assert_eq!(m[0].get("images"), Some(&json::Value::Arr(vec![json::Value::Str("QUJD".into())])));
            }
            other => panic!("bad messages: {:?}", other),
        }
        let no_img = ollama_chat_request_body("m", "t", 0.1, 8192, 64, None, false);
        assert!(!no_img.contains("images") && !no_img.contains("\"format\""));

        let r = parse_ollama_chat_response(r#"{"message":{"content":"abc"},"done_reason":"length"}"#).unwrap();
        assert_eq!(r, GemmaReply { text: "abc".into(), finish_reason: Some("length".into()) });
        assert!(gemma_truncated(&r));
    }

    #[test]
    fn truncation_heuristic_without_finish_reason() {
        let mk = |t: &str| GemmaReply { text: t.into(), finish_reason: None };
        assert!(gemma_truncated(&mk("{\"scene_prompt\":\"s\",\"char")));
        assert!(!gemma_truncated(&mk(OK_JSON)));
        assert!(!gemma_truncated(&mk(&format!("```json\n{}\n```", OK_JSON))));
    }

    #[test]
    fn payload_text_is_valid_json_after_the_instruction() {
        let p = build_payload(WD, MD1, MD2, Mode::SceneShape, None, false);
        let text = payload_to_text(&p);
        let (head, body) = text.split_once("\n\nINPUT:\n").unwrap();
        assert_eq!(head, Mode::SceneShape.llm_instruction());
        let v = json::parse(body).expect("INPUT must be valid JSON");
        assert_eq!(v.get("pass_1_tags"), Some(&json::Value::Str(p.pass_1_tags.clone())));
        assert_eq!(v.get("pass_2_tags"), Some(&json::Value::Str(p.pass_2_tags.clone())));
    }

    #[test]
    fn json_parser_basics() {
        use json::Value::*;
        assert_eq!(json::parse(r#"{"a":[1,true,null,"xé😀"]}"#).unwrap().get("a"),
            Some(&Arr(vec![Num(1.0), Bool(true), Null, Str("x\u{e9}\u{1F600}".into())])));
        assert!(json::parse("{\"a\":1} trailing").is_err());
        assert!(json::parse("{\"a\":").is_err());
        assert_eq!(json::parse(r#"{"a":1,"a":2}"#).unwrap().get("a"), Some(&Num(2.0)));
    }

    #[test]
    fn vocab_csv_loading() {
        let csv = "tag_id,name,category,count\n1,1girl,0,100\n2,\"weird,name\",0,5\n3,long_hair,0,9\n";
        let v = load_vocab_from_csv(csv);
        assert!(v.contains("1girl") && v.contains("long hair") && v.contains("weird,name"));
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn zone_header_stripping() {
        assert_eq!(flatten_zones("1. **Scene & Lighting**: soft light\n2) Camera - low angle"), "soft light, low angle");
        assert_eq!(flatten_zones("1. A: x\n\n2. B: y\nno header line"), "x, y, no header line");
    }

    #[test]
    fn subject_tags() {
        for t in ["1girl", "2boys", "6+girls", "solo", "multiple girls", "1other"] {
            assert!(is_subject_tag(t), "{}", t);
        }
        for t in ["girl", "1 girl", "solo focus", "1girls x", ""] {
            assert!(!is_subject_tag(t), "{}", t);
        }
    }

    #[test]
    fn crop_tag_policy_rejects_artifacts_ratings_sizes_counts() {
        let crop = vec![
            ("close-up".to_string(), 0.95),
            ("portrait".to_string(), 0.90),
            ("solo focus".to_string(), 0.88),
            ("breast focus".to_string(), 0.85),
            ("head out of frame".to_string(), 0.80),
            ("cropped legs".to_string(), 0.75),
            ("out of frame".to_string(), 0.90),
            ("feet out of frame".to_string(), 0.65),
            ("partially submerged".to_string(), 0.60),
            ("general".to_string(), 0.99),
            ("sensitive".to_string(), 0.95),
            ("1girl".to_string(), 0.98),
            ("large breasts".to_string(), 0.85),
            ("pyra (xenoblade)".to_string(), 0.95),
            ("silver brooch".to_string(), 0.55),
            ("hair ornament".to_string(), 0.54), // below 0.55
        ];
        let char_names = vec!["pyra (xenoblade)".to_string()];
        let filtered = filter_crop_tags(&crop, CropType::CharacterZone, &char_names, 0.55, 0.40);
        let names: Vec<String> = filtered.into_iter().map(|(t, _)| t).collect();
        assert_eq!(names, vec!["silver brooch"]);
    }

    #[test]
    fn crop_tag_policy_quadrant_concrete_objects() {
        let quad = vec![
            ("potted plant".to_string(), 0.45),
            ("bookshelf".to_string(), 0.40),
            ("simple background".to_string(), 0.90),
            ("white background".to_string(), 0.85),
            ("solo focus".to_string(), 0.70),
            ("close-up".to_string(), 0.75),
            ("wooden chair".to_string(), 0.50),
        ];
        let filtered = filter_crop_tags(&quad, CropType::Quadrant, &[], 0.55, 0.40);
        let names: Vec<String> = filtered.into_iter().map(|(t, _)| t).collect();
        assert!(names.contains(&"potted plant".to_string()));
        assert!(names.contains(&"bookshelf".to_string()));
        assert!(names.contains(&"wooden chair".to_string()));
        assert!(!names.contains(&"simple background".to_string()));
        assert!(!names.contains(&"white background".to_string()));
        assert!(!names.contains(&"solo focus".to_string()));
        assert!(!names.contains(&"close-up".to_string()));
    }

    #[test]
    fn merge_crop_tags_takes_max_score() {
        let base = vec![("blue dress".to_string(), 0.60), ("black boots".to_string(), 0.70)];
        let crop = vec![("blue dress".to_string(), 0.85), ("silver necklace".to_string(), 0.80)];
        let merged = merge_crop_tags(&base, &crop, CropType::CharacterZone, &[], 0.55, 0.40);
        let m_map: HashMap<String, f32> = merged.into_iter().collect();
        assert_eq!(m_map.get("blue dress"), Some(&0.85));
        assert_eq!(m_map.get("black boots"), Some(&0.70));
        assert_eq!(m_map.get("silver necklace"), Some(&0.80));
    }

    #[test]
    fn pure_zone_boxes_geometry() {
        // Full body: 1000x1000 image, box height = 850 (ratio = 0.85)
        let boxes = compute_zone_boxes(1000.0, 1000.0, 200.0, 100.0, 500.0, 950.0);
        let names: Vec<&str> = boxes.iter().map(|b| b.name).collect();
        assert_eq!(names, vec!["Upper", "Middle", "Lower"]);

        // Half body touching bottom: box height = 500 (ratio = 0.50, touches bottom >= 980)
        let half = compute_zone_boxes(1000.0, 1000.0, 200.0, 500.0, 500.0, 1000.0);
        let half_names: Vec<&str> = half.iter().map(|b| b.name).collect();
        assert_eq!(half_names, vec!["Upper", "Middle"]);

        // Head/bust: height = 150 (ratio = 0.15)
        let bust = compute_zone_boxes(1000.0, 1000.0, 300.0, 100.0, 500.0, 250.0);
        let bust_names: Vec<&str> = bust.iter().map(|b| b.name).collect();
        assert_eq!(bust_names, vec!["Upper"]);
    }
}
