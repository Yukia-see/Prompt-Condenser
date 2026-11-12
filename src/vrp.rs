//! LuCiD VRP regions: the boxes the region editor shows and Copy VRP regions writes.

use super::*;

/// The VRP regions for `n` characters and the canvas's source size: the edited boxes when there's one per
/// character, else the detector boxes of the last Split Image, else even strips.
pub(crate) fn vrp_current(st: &AppState, n: usize) -> (Vec<[f64; 4]>, Option<(u32, u32)>) {
    let split = st.two_tags.as_ref();
    let size = split
        .map(|s| (s.image.width(), s.image.height()))
        .or_else(|| st.image_path.as_ref().and_then(|p| image::image_dimensions(p).ok()));
    if let Some(edited) = st.vrp_regions.as_ref().filter(|v| v.len() == n) {
        return (edited.clone(), size);
    }
    let boxes = split.and_then(|s| s.boxes.as_deref());
    (two::vrp_regions(n, boxes, size), size)
}

/// The Copy VRP regions JSON for `n` characters (each box with its LoRA box in front) and where the boxes
/// came from: "edited" (region editor), "detector" (last Split Image) or "strips".
pub(crate) fn vrp_json(ui: &AppWindow, st: &AppState, n: usize) -> (String, &'static str) {
    let config = scan_config(ui);
    let prompts: Vec<String> = (0..n)
        .map(|i| {
            let lora = config.lora_of(i).trim().trim_matches(',').trim().to_string();
            let text = char_box(ui, i);
            if lora.is_empty() { text } else { format!("{lora}, {text}") }
        })
        .collect();
    let source = if st.vrp_regions.as_ref().is_some_and(|v| v.len() == n) {
        "edited"
    } else if st.two_tags.as_ref().and_then(|s| s.boxes.as_ref()).is_some() {
        "detector"
    } else {
        "strips"
    };
    let (regions, size) = vrp_current(st, n);
    (two::vrp_state_json_from(&prompts, &regions, size), source)
}

/// Shows the edited regions in the region editor.
pub(crate) fn push_vrp_regions(st: &AppState) {
    let rows: Vec<VrpRegion> = st
        .vrp_regions
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|r| VrpRegion { x: r[0] as f32, y: r[1] as f32, w: r[2] as f32, h: r[3] as f32 })
        .collect();
    let model = &st.vrp_model;
    if model.row_count() == rows.len() {
        for (i, r) in rows.into_iter().enumerate() {
            if model.row_data(i).as_ref() != Some(&r) {
                model.set_row_data(i, r);
            }
        }
    } else {
        model.set_vec(rows);
    }
}

/// Wires the region editor (move / resize) and Copy VRP regions.
pub(crate) fn wire_vrp(ui: &AppWindow, state: &State) {
    // --- Region editor (2-4 Characters tabs): move / resize each character's VRP box ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_edit_vrp_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let n = tab_chars(&ui);
            let (regions, size) = vrp_current(&state.borrow(), n);
            state.borrow_mut().vrp_regions = Some(regions);
            let (w, h) = size.unwrap_or((1216, 832));
            ui.set_vrp_aspect(w as f32 / h.max(1) as f32);
            ui.set_vrp_has_image(ui.get_image_loaded());
            push_vrp_regions(&state.borrow());
            ui.set_vrp_open(true);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_vrp_region_moved(move |i, dx, dy| {
            let Some(ui) = weak.upgrade() else { return };
            if let Some(r) = state.borrow_mut().vrp_regions.as_mut().and_then(|v| v.get_mut(i as usize)) {
                r[0] = (r[0] + dx as f64).clamp(0.0, 1.0 - r[2]);
                r[1] = (r[1] + dy as f64).clamp(0.0, 1.0 - r[3]);
            }
            push_vrp_regions(&state.borrow());
            push_vrp_preview(&ui, &state.borrow(), tab_chars(&ui));
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_vrp_region_resized(move |i, dw, dh| {
            let Some(ui) = weak.upgrade() else { return };
            if let Some(r) = state.borrow_mut().vrp_regions.as_mut().and_then(|v| v.get_mut(i as usize)) {
                r[2] = (r[2] + dw as f64).clamp(0.04, 1.0 - r[0]);
                r[3] = (r[3] + dh as f64).clamp(0.04, 1.0 - r[1]);
            }
            push_vrp_regions(&state.borrow());
            push_vrp_preview(&ui, &state.borrow(), tab_chars(&ui));
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_vrp_reset_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let n = tab_chars(&ui);
            state.borrow_mut().vrp_regions = None;
            let (regions, _) = vrp_current(&state.borrow(), n);
            state.borrow_mut().vrp_regions = Some(regions);
            push_vrp_regions(&state.borrow());
            push_vrp_preview(&ui, &state.borrow(), tab_chars(&ui));
        });
    }

    // --- LuCiD VRP regions to the clipboard (2-4 Characters tabs) ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_copy_vrp_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let n = tab_chars(&ui);
            let json = vrp_json(&ui, &state.borrow(), n).0;
            let mut st = state.borrow_mut();
            if st.clipboard.is_none() {
                st.clipboard = arboard::Clipboard::new().ok();
            }
            let msg = match st.clipboard.as_mut().map(|c| c.set_text(json)) {
                Some(Ok(())) => "VRP regions copied: paste them into the Anima VRP (core) node's vrp_state, and the Scene box into its global prompt.".to_string(),
                Some(Err(e)) => format!("Copy failed: {e}"),
                None => "Copy failed: no clipboard available.".to_string(),
            };
            ui.set_status_message(msg.into());
        });
    }
}
