//! Headless window smoke test: builds `AppWindow` on Slint's testing backend with a software
//! renderer and renders every tab and overlay, so a layout loop (Slint's "Recursion detected"
//! panic) or a broken binding fails `cargo test` instead of the first launch.

use crate::{AppWindow, ChangeCardItem, ScenePanelChar, VrpRegion};
use slint::{ComponentHandle, Model, ModelRc, VecModel};
use std::rc::Rc;

/// Renders the window once; layout, bindings and every visible item are evaluated.
fn render(ui: &AppWindow, what: &str) {
    // (the crate-root wrapper disappears when another crate turns on its `internal` feature)
    i_slint_backend_testing::testing_backend::mock_elapsed_time(300);
    if std::env::var("SMOKE_DEBUG").is_ok() { eprintln!("{what}: scroll {} overflow {}", ui.get_workspace_scroll_y(), ui.get_workspace_overflow()); }
    let shot = ui.window().take_snapshot().unwrap_or_else(|e| panic!("{what}: snapshot failed: {e}"));
    assert!(shot.width() > 0 && shot.height() > 0, "{what}: empty snapshot");
    // `SMOKE_SNAPSHOTS=<dir>` saves every state as a PNG, to look at the layout without a window
    if let Ok(dir) = std::env::var("SMOKE_SNAPSHOTS") {
        let name: String = what.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        let img = image::RgbaImage::from_raw(shot.width(), shot.height(), shot.as_bytes().to_vec()).unwrap();
        img.save(std::path::Path::new(&dir).join(format!("{name}.png"))).unwrap();
    }
}

/// Toolbar buttons that must stay on screen at every window size (the minimum width is 800 px).
const MUST_REACH: [&str; 21] = [
    "▸ Options", "▾ Options", "⚙ Settings", "Scene Panel", "↻", "⛶", "+ Add", "✦", "📋 Copy", "🗺", "✎", "BREAK Sep",
    "Dynamic", "Clean Bloat", "🕒", "🎲 Roll", "🎨", "⚡ Queue", "✕ Clear", "+ Quality", "💡",
];

/// Every instantiated button whose label starts with one of `MUST_REACH` ends inside the window: its right
/// edge always, its bottom edge too, or (inside the workspace, which scrolls when the window is too short)
/// once the workspace is scrolled to its end. The status bar's buttons never scroll, so they must fit as is.
fn assert_reachable(ui: &AppWindow, what: &str) -> usize {
    use i_slint_backend_testing::ElementQuery;
    let width = ui.window().size().width as f32 / ui.window().scale_factor();
    let height = ui.window().size().height as f32 / ui.window().scale_factor();
    let query = || {
        ElementQuery::from_root(ui)
            .match_descendants()
            .match_predicate(|e| e.accessible_label().is_some_and(|l| MUST_REACH.iter().any(|m| l.starts_with(m))))
            .find_all()
    };
    let found = query();
    if std::env::var("SMOKE_DEBUG").is_ok() {
        let all = ElementQuery::from_root(ui).match_descendants().find_all();
        let labels: Vec<String> = all.iter().filter_map(|e| e.accessible_label().map(|l| l.to_string())).collect();
        eprintln!("{what}: {} elements, labels {:?}", all.len(), &labels[..labels.len().min(40)]);
    }
    let mut below: Vec<String> = Vec::new();
    for e in &found {
        let right = e.absolute_position().x + e.size().width;
        assert!(right <= width + 0.5, "{what}: '{}' ends at {right} px, past the window's {width} px", e.accessible_label().unwrap());
        if e.absolute_position().y + e.size().height > height + 0.5 {
            below.push(e.accessible_label().unwrap().to_string());
        }
    }
    if !below.is_empty() {
        ui.set_workspace_scroll_y(-ui.get_workspace_overflow());
        let after = query();
        for label in &below {
            let ok = after.iter().any(|e| {
                let y = e.absolute_position().y;
                e.accessible_label().as_deref() == Some(label.as_str()) && y >= -0.5 && y + e.size().height <= height + 0.5
            });
            let ends: Vec<f32> = after.iter().filter(|e| e.accessible_label().as_deref() == Some(label.as_str())).map(|e| e.absolute_position().y + e.size().height).collect();
            assert!(ok, "{what}: '{label}' is below the window's {height} px even with the workspace scrolled to its end (ends at {ends:?})");
        }
        ui.set_workspace_scroll_y(0.0);
    }
    found.len()
}

#[test]
fn every_tab_and_overlay_renders_headless() {
    // the platform is per thread here (no event loop), so this test owns it
    slint::platform::set_platform(Box::new(i_slint_backend_testing::TestingBackend::new(
        i_slint_backend_testing::TestingBackendOptions {
            mock_time: true,
            threading: false,
            renderer_name: Some("software".into()),
        },
    )))
    .expect("platform already set on this thread");
    let ui = AppWindow::new().expect("AppWindow builds");
    ui.show().expect("window shows");

    // some content, so the cards strip, the scene panel and the region editor have rows to lay out
    ui.set_raw_prompt("masterpiece, 1girl, red hair, close-up, full body, beach".into());
    ui.set_change_cards(ModelRc::new(VecModel::from(vec![
        ChangeCardItem {
            id: 1,
            badge: "Framing".into(),
            badge_kind: 0,
            original: "full body".into(),
            reason: "conflicts with close-up".into(),
            suggested_alternative: "upper body".into(),
            framing_alternative: "cowboy shot".into(),
            has_framing_alt: true,
            mode: 0,
            ..Default::default()
        },
        ChangeCardItem { id: 2, badge: "Weight".into(), badge_kind: 2, original: "(red hair:1.8)".into(), ..Default::default() },
        // a 💡 Ideas card with tag chips, and the folded small fixes
        ChangeCardItem {
            id: 60_000,
            badge: "AI IDEA".into(),
            badge_kind: 3,
            reason: "Idea for the lighting".into(),
            suggested_alternative: "sunset, backlighting".into(),
            mode: 2,
            chips: ModelRc::new(VecModel::from(vec![
                crate::CardChip { tag: "sunset".into(), on: true },
                crate::CardChip { tag: "backlighting".into(), on: false },
            ])),
            ..Default::default()
        },
        ChangeCardItem { id: -1, badge: "SMALL FIXES".into(), badge_kind: 2, reason: "3 small fixes".into(), summary_count: 3, ..Default::default() },
    ])));
    // the strip's tab counts (render fills them in the app): 4 cards, so the strip is open
    let counts = ModelRc::new(VecModel::from(vec![4, 1, 2, 0, 1]));
    ui.set_card_filter_counts(counts.clone());
    let chars: Vec<ScenePanelChar> = (1..=4)
        .map(|i| ScenePanelChar { title: format!("Character {i}").into(), actions: "standing".into(), has_actions: true, ..Default::default() })
        .collect();
    ui.set_scene_panel_chars(ModelRc::new(VecModel::from(chars)));
    let regions = Rc::new(VecModel::from(vec![
        VrpRegion { x: 0.0, y: 0.0, w: 0.5, h: 1.0 },
        VrpRegion { x: 0.5, y: 0.1, w: 0.45, h: 0.85 },
    ]));
    ui.set_vrp_regions(ModelRc::from(regions.clone()));
    ui.set_vrp_preview(ModelRc::from(regions.clone()));
    ui.set_negative_text("lowres, bad hands".into());
    ui.set_negative_tokens("~5 tokens".into());
    ui.set_lora_tags("<lora:style_v2:0.8>, <lora:slider:-0.5:0.6>".into());
    let weights = ModelRc::new(VecModel::from(vec![
        crate::LoraWeight { name: "style_v2".into(), weight: "0.8".into() },
        crate::LoraWeight { name: "slider".into(), weight: "-0.5".into() },
    ]));
    ui.set_lora_weights_c1(weights.clone());
    ui.set_lora_weights_c2(weights);
    ui.set_palette_rows(ModelRc::new(VecModel::from(crate::scene_panel::palette_rows())));
    let cards = ui.get_change_cards();

    for (w, h) in [(1600, 1000), (900, 640), (800, 600)] {
        ui.window().set_size(slint::PhysicalSize::new(w, h));
        crate::settings::update_narrow(&ui);
        for tab in 0..4 {
            let at = format!("{w}x{h}, tab {}", tab + 1);
            ui.set_app_tab(tab);
            render(&ui, &at);
            for expanded in [true, false] {
                ui.set_options_expanded(expanded);
                ui.set_image_advanced(expanded);
                let state = format!("{at}, options {expanded}");
                render(&ui, &state);
                assert!(assert_reachable(&ui, &state) >= 6, "{state}: toolbar buttons not found");
            }
            ui.set_scene_ref_loaded(true);
            ui.set_scene_panel_open(true);
            render(&ui, &format!("{at}, scene panel"));
            ui.set_scene_panel_open(false);
            ui.set_scene_ref_loaded(false);

            ui.set_vrp_open(true);
            render(&ui, &format!("{at}, region editor"));
            regions.set_row_data(1, VrpRegion { x: 0.4, y: 0.2, w: 0.6, h: 0.8 });
            regions.push(VrpRegion { x: 0.1, y: 0.1, w: 0.3, h: 0.3 });
            render(&ui, &format!("{at}, region editor, rows changed"));
            regions.remove(2);
            ui.set_vrp_open(false);

            ui.set_history_rows(ModelRc::new(VecModel::from(vec![
                crate::HistoryRow { label: "masterpiece, 1girl, beach".into(), when: "just now".into(), starred: true, chars: 1 },
                crate::HistoryRow { label: "2 characters".into(), when: "5 min ago".into(), starred: false, chars: 2 },
            ])));
            ui.set_history_count(2);
            ui.set_history_open(true);
            render(&ui, &format!("{at}, history"));
            ui.set_history_open(false);

            ui.set_palette_open(true);
            render(&ui, &format!("{at}, palette"));
            ui.set_palette_open(false);

            ui.set_translate_open(true);
            ui.set_translate_busy(true);
            render(&ui, &format!("{at}, Danbooru Words"));
            ui.set_translate_busy(false);
            ui.set_translate_open(false);

            // the Dynamic view with its 🎲 Roll pill and a rolled preview
            ui.set_dynamic_enabled(true);
            ui.set_after_view(2);
            ui.set_after_rolled(true);
            render(&ui, &format!("{at}, rolled preview"));
            assert_reachable(&ui, &format!("{at}, rolled preview"));
            ui.set_after_rolled(false);
            ui.set_after_view(0);
            ui.set_dynamic_enabled(false);

            // Zen mode, the negative accordion (and Flux: no negative), the clean banner and its peek
            ui.set_zen_mode(true);
            render(&ui, &format!("{at}, zen"));
            assert_reachable(&ui, &format!("{at}, zen"));
            ui.set_zen_mode(false);
            ui.set_negative_expanded(false);
            render(&ui, &format!("{at}, negative collapsed"));
            assert_reachable(&ui, &format!("{at}, negative collapsed"));
            ui.set_negative_expanded(true);
            ui.set_negative_text("".into());
            render(&ui, &format!("{at}, no negative"));
            ui.set_negative_text("lowres, bad hands".into());
            // a filter tab with no cards: the strip (and its tabs) must stay open
            ui.set_change_cards(ModelRc::new(VecModel::<ChangeCardItem>::default()));
            ui.set_card_filter(1);
            render(&ui, &format!("{at}, empty filter tab"));
            assert_reachable(&ui, &format!("{at}, empty filter tab"));
            ui.set_card_filter(0);
            ui.set_card_filter_counts(ModelRc::new(VecModel::from(vec![0, 0, 0, 0, 0])));
            ui.set_scene_summary("".into());
            render(&ui, &format!("{at}, cards collapsed"));
            ui.set_cards_peek(true);
            render(&ui, &format!("{at}, cards peek"));
            ui.set_cards_peek(false);
            ui.set_change_cards(cards.clone());
            ui.set_card_filter_counts(counts.clone());
        }
    }
    ui.hide().unwrap();
}
