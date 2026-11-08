//! Python-vs-Rust parity: tests/parity.json is generated from the Python reference
//! (tools/gen_parity.py). Every case must produce identical output in Rust.
//!
//! Test-only module. In your app add to main.rs / lib.rs:
//!     mod prompt_condenser;
//!     #[cfg(test)] mod prompt_condenser_parity;
//! and keep tests/parity.json next to your Cargo.toml.

use crate::prompt_condenser::json::{self, Value};
use crate::prompt_condenser::*;
use std::collections::HashSet;

fn s(v: &Value, key: &str) -> String {
    match v.get(key) {
        Some(Value::Str(x)) => x.clone(),
        other => panic!("missing string {}: {:?}", key, other),
    }
}

fn list(v: &Value, key: &str) -> Vec<String> {
    match v.get(key) {
        Some(Value::Arr(items)) => items
            .iter()
            .map(|i| match i {
                Value::Str(x) => x.clone(),
                other => panic!("non-string in {}: {:?}", key, other),
            })
            .collect(),
        other => panic!("missing list {}: {:?}", key, other),
    }
}

fn set(v: &Value, key: &str) -> HashSet<String> {
    list(v, key).into_iter().collect()
}

fn mode(v: &Value) -> Mode {
    Mode::parse(&s(v, "mode")).unwrap()
}

fn cases<'a>(root: &'a Value, name: &str) -> &'a Vec<Value> {
    match root.get(name) {
        Some(Value::Arr(a)) => a,
        other => panic!("missing cases {}: {:?}", name, other),
    }
}

#[test]
fn rust_matches_python_reference() {
    let root = json::parse(include_str!("../tests/parity.json")).expect("parity.json");
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;
    let mut check = |group: &str, input: String, got: String, want: String| {
        total += 1;
        if got != want {
            failures.push(format!("[{}] input: {}\n  rust:   {}\n  python: {}", group, input, got, want));
        }
    };

    for c in cases(&root, "router") {
        let (p1, p2) = sort_tag_lists(&s(c, "wd"), mode(c));
        let want = c.get("expect").unwrap();
        check(
            "router",
            format!("{:?} {:?}", s(c, "wd"), s(c, "mode")),
            format!("{} | {}", p1.join(", "), p2.join(", ")),
            format!("{} | {}", s(want, "pass1_tags"), s(want, "pass2_tags")),
        );
    }

    for c in cases(&root, "analyze") {
        let v = set(c, "vocab");
        let a = analyze_prose_with_vocab(&s(c, "text"), &v);
        let want = c.get("expect").unwrap();
        let fmt = |tags: &[String], strong: Vec<String>, weak: Vec<String>, neg: &[String]| {
            format!("{:?} {:?} {:?} {:?}", tags, strong, weak, neg)
        };
        check(
            "analyze",
            s(c, "text"),
            fmt(&a.tags, a.strong.iter().cloned().collect(), a.weak.iter().cloned().collect(), &a.negated),
            fmt(&list(want, "tags"), list(want, "strong"), list(want, "weak"), &list(want, "negated")),
        );
    }

    for c in cases(&root, "merge") {
        let v = set(c, "vocab");
        let m = merge_tags(&s(c, "wd"), &s(c, "md1"), &s(c, "md2"), mode(c), Some(&v));
        let want = c.get("expect").unwrap();
        let fmt = |a: &[String], b: &[String], sub: &[String], dm: &[String], dw: &[String], dh: &[String], n: &[String], u: &[String]| {
            format!("{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}", a, b, sub, dm, dw, dh, n, u)
        };
        check(
            "merge",
            format!("wd={:?} md1={:?} md2={:?} mode={:?} vocab={:?}", s(c, "wd"), s(c, "md1"), s(c, "md2"), s(c, "mode"), list(c, "vocab")),
            fmt(&m.pass1_tags, &m.pass2_tags, &m.subject_tags, &m.dropped_vlm, &m.dropped_wd14, &m.dropped_hedged, &m.negated_vlm, &m.unrouted_wd14),
            fmt(
                &list(want, "pass1_tags"),
                &list(want, "pass2_tags"),
                &list(want, "subject_tags"),
                &list(want, "dropped_vlm"),
                &list(want, "dropped_wd14"),
                &list(want, "dropped_hedged"),
                &list(want, "negated_vlm"),
                &list(want, "unrouted_wd14"),
            ),
        );
    }

    for c in cases(&root, "nollm") {
        let v = set(c, "vocab");
        let keep = matches!(c.get("keep"), Some(Value::Bool(true)));
        let cfg = Config { keep_unmatched_prose: keep, ..Config::default() };
        let out = process_direct(&s(c, "wd"), &s(c, "md1"), &s(c, "md2"), mode(c), Some(&v), &cfg);
        let want = c.get("expect").unwrap();
        check(
            "nollm",
            format!("wd={:?} md1={:?} md2={:?} mode={:?} keep={}", s(c, "wd"), s(c, "md1"), s(c, "md2"), s(c, "mode"), keep),
            format!("{}\n{}", out.scene_prompt, out.character_prompt),
            format!("{}\n{}", s(want, "scene_prompt"), s(want, "character_prompt")),
        );
    }

    for c in cases(&root, "parse") {
        let out = parse_gemma_json_response(&s(c, "raw"), &Config::default());
        let want = c.get("expect").unwrap();
        check(
            "parse",
            format!("{:?}", s(c, "raw")),
            format!("{:?} | {:?}", out.scene_prompt, out.character_prompt),
            format!("{:?} | {:?}", s(want, "scene_prompt"), s(want, "character_prompt")),
        );
    }

    for c in cases(&root, "flatten") {
        check("flatten", format!("{:?}", s(c, "text")), flatten_zones(&s(c, "text")), s(c, "expect"));
    }

    for c in cases(&root, "style") {
        check("style", format!("{:?}", s(c, "text")), style_tags(&s(c, "text"), true), s(c, "expect"));
    }

    for c in cases(&root, "payload") {
        let v = set(c, "vocab");
        let vision = matches!(c.get("vision"), Some(Value::Bool(true)));
        let p = build_payload(&s(c, "wd"), &s(c, "md1"), &s(c, "md2"), mode(c), Some(&v), vision);
        check(
            "payload",
            format!("wd={:?} md1={:?} md2={:?} mode={:?} vision={}", s(c, "wd"), s(c, "md1"), s(c, "md2"), s(c, "mode"), vision),
            payload_to_text(&p),
            s(c, "expect"),
        );
    }

    for c in cases(&root, "filter_crop") {
        let tags: Vec<(String, f32)> = match c.get("tags") {
            Some(Value::Arr(items)) => items
                .iter()
                .map(|item| match item {
                    Value::Arr(pair) => (
                        match &pair[0] { Value::Str(s) => s.clone(), _ => String::new() },
                        match &pair[1] { Value::Num(n) => *n as f32, _ => 0.0 },
                    ),
                    _ => (String::new(), 0.0),
                })
                .collect(),
            _ => Vec::new(),
        };
        let ctype = match s(c, "crop_type").as_str() {
            "quadrant" => CropType::Quadrant,
            _ => CropType::CharacterZone,
        };
        let cnames = list(c, "character_names");
        let min_t = match c.get("min_attribute_threshold") { Some(Value::Num(n)) => *n as f32, _ => 0.55 };
        let sc_t = match c.get("scene_object_threshold") { Some(Value::Num(n)) => *n as f32, _ => 0.40 };

        let got = filter_crop_tags(&tags, ctype, &cnames, min_t, sc_t);
        let got_fmt: Vec<String> = got.iter().map(|(t, s)| format!("{}:{:.2}", t, s)).collect();

        let want_fmt: Vec<String> = match c.get("expect") {
            Some(Value::Arr(items)) => items
                .iter()
                .map(|item| match item {
                    Value::Arr(pair) => {
                        let tag = match &pair[0] { Value::Str(s) => s.clone(), _ => String::new() };
                        let score = match &pair[1] { Value::Num(n) => *n as f32, _ => 0.0 };
                        format!("{}:{:.2}", tag, score)
                    }
                    _ => String::new(),
                })
                .collect(),
            _ => Vec::new(),
        };

        check("filter_crop", format!("{:?} {:?}", tags, ctype), got_fmt.join(", "), want_fmt.join(", "));
    }

    for c in cases(&root, "zone_boxes") {
        let img_w = match c.get("img_w") { Some(Value::Num(n)) => *n as f32, _ => 0.0 };
        let img_h = match c.get("img_h") { Some(Value::Num(n)) => *n as f32, _ => 0.0 };
        let x0 = match c.get("x0") { Some(Value::Num(n)) => *n as f32, _ => 0.0 };
        let y0 = match c.get("y0") { Some(Value::Num(n)) => *n as f32, _ => 0.0 };
        let x1 = match c.get("x1") { Some(Value::Num(n)) => *n as f32, _ => 0.0 };
        let y1 = match c.get("y1") { Some(Value::Num(n)) => *n as f32, _ => 0.0 };

        let got = compute_zone_boxes(img_w, img_h, x0, y0, x1, y1);
        let got_str: Vec<String> = got.iter().map(|b| format!("{}:{:.1},{:.1},{:.1},{:.1}", b.name, b.x0, b.y0, b.x1, b.y1)).collect();

        let want_str: Vec<String> = match c.get("expect") {
            Some(Value::Arr(items)) => items
                .iter()
                .map(|item| {
                    let name = s(item, "name");
                    let bx0 = match item.get("x0") { Some(Value::Num(n)) => *n as f32, _ => 0.0 };
                    let by0 = match item.get("y0") { Some(Value::Num(n)) => *n as f32, _ => 0.0 };
                    let bx1 = match item.get("x1") { Some(Value::Num(n)) => *n as f32, _ => 0.0 };
                    let by1 = match item.get("y1") { Some(Value::Num(n)) => *n as f32, _ => 0.0 };
                    format!("{}:{:.1},{:.1},{:.1},{:.1}", name, bx0, by0, bx1, by1)
                })
                .collect(),
            _ => Vec::new(),
        };

        check("zone_boxes", format!("{}x{} box: {},{},{},{}", img_w, img_h, x0, y0, x1, y1), got_str.join(" | "), want_str.join(" | "));
    }

    let shown: Vec<&String> = failures.iter().take(5).collect();
    assert!(
        failures.is_empty(),
        "{} of {} parity cases differ from the Python reference. First {}:\n\n{}",
        failures.len(),
        total,
        shown.len(),
        shown.iter().map(|f| f.as_str()).collect::<Vec<_>>().join("\n\n")
    );
    println!("parity: all {} cases match the Python reference", total);
}
