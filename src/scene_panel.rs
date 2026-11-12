//! Scene / Pose reference and the Scene Panel (per-character View / Angle / Pose lists).

use super::*;

pub(crate) fn scene_overrides(ui: &AppWindow) -> sceneref::Overrides {
    sceneref::Overrides {
        scene: ui.get_scene_override_setting().to_string(),
        exclude: ui.get_scene_override_exclude().to_string(),
        chars: [
            ui.get_scene_override_c1().to_string(),
            ui.get_scene_override_c2().to_string(),
            ui.get_scene_override_c3().to_string(),
            ui.get_scene_override_c4().to_string(),
        ],
    }
}

/// Stores a new reading and applies its plan across the boxes.
pub(crate) fn set_scene_reading(ui: &AppWindow, state: &State, actions: Vec<sceneref::Action>, raw_tags: String) {
    let mut st = state.borrow_mut();
    // take out the old reading first, if any
    let old_boxes = st.scene_ref.take().map(|r| r.boxes).unwrap_or_default();
    drop(st);
    restore_boxes(ui, &old_boxes);

    ui.set_scene_override_setting("".into());
    ui.set_scene_override_c1("".into());
    ui.set_scene_override_c2("".into());
    ui.set_scene_override_c3("".into());
    ui.set_scene_override_c4("".into());
    ui.set_scene_prompt("".into());

    let mut st = state.borrow_mut();
    st.scene_ref = Some(SceneRef { raw_tags, actions, boxes: Vec::new() });
    drop(st);
    apply_scene_ref(ui, state);
    sync_scene_panel(ui, state);
}

/// Computes the plan from the current actions and overrides, and injects into the boxes.
pub(crate) fn apply_scene_ref(ui: &AppWindow, state: &State) {
    let n = tab_chars(ui);
    let (actions, old_boxes) = {
        let st = state.borrow();
        let Some(sr) = st.scene_ref.as_ref() else { return };
        (sr.actions.clone(), sr.boxes.clone())
    };
    let ov = scene_overrides(ui);
    let plan = sceneref::plan(&actions, &ov, n);

    // Box index: 0 = Before / Char 1, 1 = Char 2, 2 = Char 3, 3 = Char 4, 4 = Scene box
    // (box, tags, which predicate: 0 = actions, 1 = scene, 2 = both)
    let targets: Vec<(usize, Vec<String>, u8)> = {
        let mut list = Vec::new();
        for i in 0..n {
            let tags = plan.chars.get(i).cloned().unwrap_or_default();
            list.push((i, tags, 0));
        }
        list.push((4, plan.scene.clone(), 1));
        list
    };

    let mut new_snaps = Vec::new();
    let other_prompts: Vec<String> = (0..n).map(|i| char_box(ui, i)).collect();
    for (target, refs, kind) in targets {
        let current = if target == 4 { ui.get_scene_prompt().to_string() } else { char_box(ui, target) };
        let original = match old_boxes.iter().find(|b| b.target == target && b.injected == current) {
            Some(b) => b.original.clone(),
            None => current.clone(),
        };
        let injected = if target == 4 {
            // When doing references or scene: clear shared prompt; if text already exists that is different from character prompts, add to it
            sceneref::update_shared_scene_prompt(&original, &refs, &other_prompts)
        } else {
            let body_shape_on = effective_scene_body_shape(ui);
            sceneref::inject(&original, &refs, &|t| match kind {
                0 => sceneref::is_action(t) || (body_shape_on && crate::scene::is_body_shape_tag(&t.normalized)),
                1 => scene::is_scene_or_pose(t),
                _ => sceneref::is_pose_or_scene(t),
            })
        };
        if target == 4 {
            ui.set_scene_prompt(injected.clone().into());
        } else {
            set_char_box(ui, target, &injected);
        }
        new_snaps.push(BoxSnap { target, original, injected, added: refs });
    }

    if let Some(sr) = state.borrow_mut().scene_ref.as_mut() {
        sr.boxes = new_snaps;
    }
    analyze(ui, state);
}

/// Restores boxes from a snapshot: exact match restores original, modified only strips added tags.
pub(crate) fn restore_boxes(ui: &AppWindow, snaps: &[BoxSnap]) {
    for b in snaps {
        let current = if b.target == 4 { ui.get_scene_prompt().to_string() } else { char_box(ui, b.target) };
        let restored = if current == b.injected {
            b.original.clone()
        } else {
            parser::parse_prompt(&current)
                .into_iter()
                .filter(|t| !b.added.iter().any(|x| danbooru::key(x) == t.normalized))
                .map(|t| t.raw.trim().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        if b.target == 4 {
            ui.set_scene_prompt(restored.into());
        } else {
            set_char_box(ui, b.target, &restored);
        }
    }
}

/// Syncs the horizontal Scene Panel cards in Slint from the current plan.
/// Extracts only action and pose tags describing what a character is doing.
pub(crate) fn extract_char_actions(prompt: &str) -> String {
    let tokens = parser::parse_prompt(prompt);
    tokens
        .into_iter()
        .filter(sceneref::is_action)
        .map(|t| t.raw.trim().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn sync_scene_panel(ui: &AppWindow, state: &State) {
    let n = tab_chars(ui);
    let st = state.borrow();
    let ov = scene_overrides(ui);
    let plan = st.scene_ref.as_ref().map(|sr| sceneref::plan(&sr.actions, &ov, n));

    let cur_scene = ui.get_scene_prompt().to_string();
    if !cur_scene.trim().is_empty() && ui.get_scene_override_setting().trim().is_empty() {
        ui.set_scene_override_setting(cur_scene.into());
    } else if let Some(ref p) = plan {
        if ui.get_scene_override_setting().trim().is_empty() && !p.scene.is_empty() {
            ui.set_scene_override_setting(p.scene.join(", ").into());
        }
    }

    let mut items = Vec::new();
    for i in 0..n {
        let title = if n == 1 { "Character 1".to_string() } else { format!("Character {}", i + 1) };
        let cbox = char_box(ui, i);
        let actions = if let Some(ref p) = plan {
            let acts = p.chars.get(i).map(|v| v.join(", ")).unwrap_or_default();
            if !acts.trim().is_empty() {
                acts
            } else {
                extract_char_actions(&cbox)
            }
        } else {
            extract_char_actions(&cbox)
        };
        let has_actions = !actions.trim().is_empty();

        let pose_group = profiles::group_tags(profiles::POSES);
        let cbox_tokens = parser::parse_prompt(&cbox);
        let has_dynamic_pose = cbox_tokens.iter().any(|t| {
            let n = t.normalized.as_str();
            n.starts_with('{') && n.ends_with('}') && {
                let opts: Vec<String> = n.trim_matches(|c| c == '{' || c == '}').split('|').map(|o| o.trim().to_string()).collect();
                opts.iter().any(|o| pose_group.contains(o))
            }
        });
        let pose_index = if has_dynamic_pose {
            2 // Dynamic / Random Pose
        } else {
            let idx = list_index_in(&cbox, profiles::POSES);
            if idx > 0 && profiles::POSES[idx].0 != "Random pose (dynamic prompt)" {
                (idx + 2) as i32
            } else {
                0
            }
        };

        let view_index = match list_index_in(&cbox, profiles::VIEWS) {
            0 => 0,
            i => i + 1,
        } as i32;
        let angle_index = match list_index_in(&cbox, profiles::CAMERA_ANGLES) {
            0 if cbox.contains(profiles::CAMERA_ANGLES[profiles::CAMERA_ANGLES.len() - 1].1) => profiles::CAMERA_ANGLES.len() as i32,
            0 => 0,
            i => (i + 1) as i32,
        };
        items.push(ScenePanelChar {
            title: title.into(),
            actions: actions.into(),
            has_actions,
            view_index,
            angle_index,
            pose_index,
        });
    }
    ui.set_scene_panel_chars(Rc::new(VecModel::from(items)).into());
}

/// Puts `new_tags` from one dropdown list (VIEWS / CAMERA_ANGLES) into a character's
/// prompt, taking out that list's other tags first (and its {a|b} groups), and keeping
/// everything else, including other dropdowns' tags.
pub(crate) fn swap_group(current: &str, new_tags: &str, list: &[(&str, &str)]) -> String {
    let group = profiles::group_tags(list);
    let chosen: Vec<String> = parser::parse_prompt(new_tags).into_iter().map(|t| t.normalized).collect();
    let of_group = |t: &parser::PromptToken| {
        let n = t.normalized.as_str();
        if n.starts_with('{') && n.ends_with('}') {
            // a dynamic group of this list's options
            let opts: Vec<String> = n.trim_matches(|c| c == '{' || c == '}').split('|').map(|o| o.trim().to_string()).collect();
            return !opts.is_empty() && opts.iter().any(|o| group.contains(o));
        }
        group.iter().any(|g| g == n) && !chosen.iter().any(|c| c == n)
    };
    let mut kept: Vec<String> = parser::parse_prompt(current)
        .into_iter()
        .filter(|t| !of_group(t))
        .map(|t| t.raw.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    for t in parser::parse_prompt(new_tags) {
        if !kept.iter().any(|k| danbooru::key(k) == t.normalized) {
            kept.push(t.raw.trim().to_string());
        }
    }
    kept.join(", ")
}

/// Which entry of a dropdown list (index into `list`) a prompt already uses: the first
/// whose tags are all in it, or the entry with the most matching tags. 0 = none.
pub(crate) fn list_index_in(prompt: &str, list: &[(&str, &str)]) -> usize {
    let toks: Vec<String> = parser::parse_prompt(prompt).into_iter().map(|t| t.normalized).collect();
    // First, look for an entry where all tags match (most specific first)
    let mut best_all: Option<(usize, usize)> = None;
    for (i, (_, tags)) in list.iter().enumerate().skip(1) {
        if tags.starts_with('{') {
            if toks.iter().any(|tok| tok == *tags) {
                return i;
            }
            continue;
        }
        let want: Vec<String> = parser::parse_prompt(tags).into_iter().map(|t| t.normalized).collect();
        if !want.is_empty()
            && want.iter().all(|w| toks.contains(w))
            && best_all.map(|(_, count)| want.len() > count).unwrap_or(true)
        {
            best_all = Some((i, want.len()));
        }
    }
    if let Some((i, _)) = best_all {
        return i;
    }
    // Otherwise, find the entry with the most matching tags (at least one)
    let mut best_any: Option<(usize, usize)> = None;
    for (i, (_, tags)) in list.iter().enumerate().skip(1) {
        if tags.starts_with('{') {
            continue;
        }
        let want: Vec<String> = parser::parse_prompt(tags).into_iter().map(|t| t.normalized).collect();
        let matches = want.iter().filter(|w| toks.contains(w)).count();
        if matches > 0 && best_any.map(|(_, count)| matches > count).unwrap_or(true) {
            best_any = Some((i, matches));
        }
    }
    best_any.map(|(i, _)| i).unwrap_or(0)
}

/// A Scene Panel View / Angle dropdown pick for character `idx`. `auto` takes that
/// list's tags from the Scene / Pose reference.
pub(crate) fn scene_list_selected(ui: &AppWindow, state: &State, idx: usize, val: &str, list: &[(&str, &str)], what: &str) {
    if val.starts_with('+') || val.is_empty() {
        return;
    }
    let current = char_box(ui, idx);
    let tags = if val.starts_with('⚡') {
        let group = profiles::group_tags(list);
        let found: Vec<String> = state
            .borrow()
            .scene_ref
            .as_ref()
            .map(|r| {
                r.actions
                    .iter()
                    .filter(|a| matches!(a.owner, sceneref::Owner::Scene | sceneref::Owner::All) || a.owner == sceneref::Owner::Char(idx))
                    .filter(|a| group.contains(&danbooru::key(&a.tag)))
                    .map(|a| a.tag.clone())
                    .collect()
            })
            .unwrap_or_default();
        if found.is_empty() {
            ui.set_scene_ref_status(format!("No {what} found in the Scene reference for Character {}.", idx + 1).into());
            return;
        }
        found.join(", ")
    } else {
        match list.iter().find(|(l, _)| *l == val) {
            Some((_, t)) if !t.is_empty() => t.to_string(),
            _ => return,
        }
    };
    set_char_box(ui, idx, &swap_group(&current, &tags, list));
    sync_scene_panel(ui, state);
    analyze(ui, state);
    ui.set_scene_ref_status(format!("Character {} {what}: {val}", idx + 1).into());
}

/// Injects or replaces a pose for a character prompt, replacing existing pose tags from the standard poses list.
pub(crate) fn inject_pose_into_prompt(current: &str, new_pose: &str) -> String {
    swap_group(current, new_pose, profiles::POSES)
}

/// Removes the reference image completely.
pub(crate) fn clear_scene_ref(ui: &AppWindow, state: &State) {
    state.borrow_mut().scene_ref_path = None;
    state.borrow_mut().image_gens.scene += 1;
    let old = state.borrow_mut().scene_ref.take();
    if let Some(r) = old {
        restore_boxes(ui, &r.boxes);
    }
    ui.set_scene_ref_loaded(false);
    ui.set_scene_ref_thumb(slint::Image::default());
    ui.set_scene_ref_status("".into());
    ui.set_scene_panel_open(false);
    sync_scene_panel(ui, state);
    update_pending(ui, state);
    analyze(ui, state);
}

/// Wires reference use transfer, the Scene / Pose reference image and the Scene Panel.
/// The 🎨 Palette drawer's rows: each `scene::PALETTE` group in rows of at most this many chips (shown
/// without the escapes; the click handler maps a chip back to its escaped tag).
const PALETTE_PER_ROW: usize = 5;

pub(crate) fn palette_rows() -> Vec<PaletteRow> {
    let mut rows = Vec::new();
    for (title, tags) in scene::PALETTE {
        for (i, chunk) in tags.chunks(PALETTE_PER_ROW).enumerate() {
            rows.push(PaletteRow {
                title: if i == 0 { (*title).into() } else { SharedString::new() },
                tags: ModelRc::new(VecModel::from(chunk.iter().map(|t| SharedString::from(t.replace('\\', ""))).collect::<Vec<_>>())),
            });
        }
    }
    rows
}

pub(crate) fn wire_scene_panel(ui: &AppWindow, state: &State) {
    // --- 🎨 Palette: an explicit click appends one lighting / atmosphere / style tag ---
    ui.set_palette_rows(ModelRc::new(VecModel::from(palette_rows())));
    {
        let weak = ui.as_weak();
        ui.on_palette_tag_clicked(move |tag| {
            let Some(ui) = weak.upgrade() else { return };
            let tag = scene::PALETTE.iter().flat_map(|(_, tags)| tags.iter()).find(|t| t.replace('\\', "") == tag.as_str()).map_or(tag.to_string(), |t| t.to_string());
            let scene_box = ui.get_app_tab() >= 1;
            let text = if scene_box { ui.get_scene_prompt() } else { ui.get_raw_prompt() };
            let target = if scene_box { "the Scene box" } else { "Before" };
            match scene::append_palette_tag(&text, &tag) {
                Some(t) => {
                    if scene_box {
                        ui.set_scene_prompt(t.into());
                    } else {
                        ui.set_raw_prompt(t.into());
                    }
                    ui.set_status_message(format!("Added {tag} to {target}.").into());
                    ui.invoke_analyze_requested();
                }
                None => ui.set_status_message(format!("{tag} is already in {target}.").into()),
            }
        });
    }

    // --- Reference use presets: Identity / Outfit / Atmosphere transfer (1 Character tab) ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_transfer_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let Some(preset) = scene::Transfer::from_index(ui.get_reference_use_index()) else { return };
            let Some(path) = state.borrow().scene_ref_path.clone() else {
                ui.set_scene_ref_status("Load a reference image first.".into());
                return;
            };
            if ui.get_raw_prompt().trim().is_empty() {
                ui.set_scene_ref_status("Read or type your character first (Before), then transfer.".into());
                return;
            }
            let gen = (state.borrow().image_gens, ImageSlots::Scene);
            ai_engine::reset_cancel();
            ui.set_tagger_busy(true);
            ui.set_scene_ref_status("Reading the reference with the fast tagger…".into());
            let weak = ui.as_weak();
            std::thread::spawn(move || {
                let started = std::time::Instant::now();
                // the fast Tagger mode reading (one pass + cleanup rules + picture checks), ~5 s
                let read = twoimg::tag_one_passes(&path, &|_| (), false, false)
                    .map(|scores| ai_engine::clean_tagger_prompt(&twoimg::build_fast_prompt(&scores, 0.0).prompt, &path));
                let secs = started.elapsed().as_secs_f32();
                let _ = weak.upgrade_in_event_loop(move |ui| {
                    ui.set_tagger_busy(false);
                    if image_changed_since(&ui, gen) {
                        return;
                    }
                    match read {
                        Ok(reference) => with_state(|state| {
                            let before = ui.get_raw_prompt().to_string();
                            let merged = scene::transfer(&before, &reference, preset);
                            // Undo goes back to Before as it was right before this transfer
                            state.borrow_mut().prompt_before_image = Some(before);
                            ui.set_image_can_undo(true);
                            ui.set_raw_prompt(merged.into());
                            analyze(&ui, state);
                            let what = match preset {
                                scene::Transfer::Identity => "body, outfit, pose and scene",
                                scene::Transfer::Outfit => "outfit",
                                scene::Transfer::Atmosphere => "setting, light and style",
                            };
                            // a reference with several people mixes their outfits / bodies
                            let group = reference.split(',').map(danbooru::key).any(|k| {
                                two::count_tag(&k).is_some_and(|(n, _)| n >= 2) || k == "multiple girls" || k == "multiple boys"
                            });
                            let note = if group && preset != scene::Transfer::Atmosphere {
                                " • the reference shows several people, so their tags are combined"
                            } else {
                                ""
                            };
                            ui.set_scene_ref_status(format!("Took the reference's {what} • {secs:.1} s (Undo restores Before){note}").into());
                        }),
                        Err(e) => ui.set_scene_ref_status(format!("Reading the reference failed: {e:#}").into()),
                    }
                });
            });
        });
    }

    // --- Scene / Pose reference image ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_load_scene_ref_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let Some(path) = rfd::FileDialog::new()
                .set_title("Pick a reference image for the pose and the scene")
                .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
                .pick_file()
            else {
                return;
            };
            if let Ok(img) = slint::Image::load_from_path(&path) {
                ui.set_scene_ref_thumb(img);
            }
            ui.set_scene_ref_loaded(true);
            // a new reference replaces the old one: its tags come out now, the new
            // one's go in on Read
            let old_boxes = {
                let mut st = state.borrow_mut();
                st.scene_ref_path = Some(path);
                st.image_gens.scene += 1;
                st.scene_ref.take().map(|r| r.boxes).unwrap_or_default()
            };
            restore_boxes(&ui, &old_boxes);
            ui.set_scene_prompt("".into());
            analyze(&ui, &state);
            ui.set_scene_ref_status("Scene image loaded. Click '⚡ Auto Scene (Fast)' or '🧠 Read (AI)'.".into());
            update_pending(&ui, &state);
            sync_scene_panel(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_remove_scene_ref_requested(move || {
            if let Some(ui) = weak.upgrade() {
                clear_scene_ref(&ui, &state);
            }
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_char_image_requested(move |which| {
            let Some(ui) = weak.upgrade() else { return };
            let which = (which.max(0) as usize).min(tab_chars(&ui) - 1);
            let Some(path) = rfd::FileDialog::new()
                .set_title(format!("Pick a reference image for character {}", which + 1))
                .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
                .pick_file()
            else {
                return;
            };
            if let Ok(img) = slint::Image::load_from_path(&path) {
                match which {
                    0 => { ui.set_char1_image_thumb(img.clone()); ui.set_char1_image_loaded(true); }
                    1 => { ui.set_char2_image_thumb(img.clone()); ui.set_char2_image_loaded(true); }
                    2 => { ui.set_char3_image_thumb(img.clone()); ui.set_char3_image_loaded(true); }
                    _ => { ui.set_char4_image_thumb(img.clone()); ui.set_char4_image_loaded(true); }
                }
                state.borrow_mut().char_thumbs[which] = Some(img);
            }
            {
                let mut st = state.borrow_mut();
                st.char_paths[which] = Some(path);
                st.image_gens.chars[which] += 1;
                st.char_images[which] = None;
            }
            update_pending(&ui, &state);
            ui.set_image_status(format!("Character {} image loaded. Click Read when ready.", which + 1).into());
        });
    }
    // --- Scene Panel callbacks: Auto Scene, Add Action, Overrides ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_auto_scene_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            // no scene image yet: pick one, then scan it right away (one click, not two)
            if state.borrow().scene_ref_path.is_none() {
                ui.invoke_load_scene_ref_requested();
                if state.borrow().scene_ref_path.is_none() {
                    return; // the picker was cancelled
                }
            }
            run_scene_fast_scan(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_overrides_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            let setting = ui.get_scene_override_setting().to_string();
            let cleaned = twoimg::clean_list(&setting, true);
            ui.set_scene_prompt(cleaned.into());
            apply_scene_ref(&ui, &state);
            analyze(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_view_selected({
            let weak = weak.clone();
            let state = state.clone();
            move |which, val| {
                if let Some(ui) = weak.upgrade() {
                    scene_list_selected(&ui, &state, which.max(0) as usize, &val, profiles::VIEWS, "view");
                }
            }
        });
        ui.on_scene_angle_selected({
            let weak = weak.clone();
            let state = state.clone();
            move |which, val| {
                if let Some(ui) = weak.upgrade() {
                    scene_list_selected(&ui, &state, which.max(0) as usize, &val, profiles::CAMERA_ANGLES, "camera angle");
                }
            }
        });
        ui.on_scene_pose_selected(move |which, val| {
            let Some(ui) = weak.upgrade() else { return };
            let idx = which.max(0) as usize;
            let val_str = val.to_string();

            if val_str == "+ Add Pose ▾" || val_str.is_empty() {
                return;
            }

            let current = char_box(&ui, idx);

            let new_prompt = if val_str == "⚡ Auto (from Scene Ref)" {
                let group = profiles::group_tags(profiles::POSES);
                let detected: Vec<String> = {
                    let st = state.borrow();
                    if let Some(scene) = st.scene_ref.as_ref() {
                        scene.actions.iter()
                            .filter(|a| matches!(a.owner, sceneref::Owner::Char(c) if c == idx) || matches!(a.owner, sceneref::Owner::All))
                            .filter(|a| group.contains(&danbooru::key(&a.tag)))
                            .map(|a| a.tag.clone())
                            .collect()
                    } else {
                        Vec::new()
                    }
                };
                if detected.is_empty() {
                    ui.set_scene_ref_status(format!("No pose detected in Scene reference for Character {}.", idx + 1).into());
                    return;
                }
                inject_pose_into_prompt(&current, &detected.join(", "))
            } else if val_str == "🎲 Dynamic / Random Pose" {
                inject_pose_into_prompt(&current, "{standing|sitting|leaning against wall|dynamic pose|looking back|arms up}")
            } else {
                let pose_tag = profiles::POSES.iter()
                    .find(|(lbl, _)| *lbl == val_str)
                    .map(|(_, tags)| *tags)
                    .unwrap_or(&val_str);
                if pose_tag.is_empty() {
                    return;
                }
                inject_pose_into_prompt(&current, pose_tag)
            };

            set_char_box(&ui, idx, &new_prompt);
            sync_scene_panel(&ui, &state);
            analyze(&ui, &state);
            ui.set_scene_ref_status(format!("Character {} pose: {}", idx + 1, val_str).into());
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_action_add_requested(move |which| {
            let Some(ui) = weak.upgrade() else { return };
            let idx = which.max(0) as usize;
            let current = char_box(&ui, idx);

            // 1. Check if a scene reference detected poses for this character
            let group = profiles::group_tags(profiles::POSES);
            let detected: Vec<String> = {
                let st = state.borrow();
                if let Some(scene) = st.scene_ref.as_ref() {
                    scene.actions.iter()
                        .filter(|a| matches!(a.owner, sceneref::Owner::Char(c) if c == idx) || matches!(a.owner, sceneref::Owner::All))
                        .filter(|a| group.contains(&danbooru::key(&a.tag)))
                        .map(|a| a.tag.clone())
                        .collect()
                } else {
                    Vec::new()
                }
            };

            let new_prompt = if !detected.is_empty() {
                inject_pose_into_prompt(&current, &detected.join(", "))
            } else {
                inject_pose_into_prompt(&current, "standing")
            };

            set_char_box(&ui, idx, &new_prompt);
            sync_scene_panel(&ui, &state);
            analyze(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_char_clear_requested(move |which| {
            let Some(ui) = weak.upgrade() else { return };
            let idx = which.max(0) as usize;
            let current = char_box(&ui, idx);
            let new_prompt = sceneref::replace(&current, &[], &|t| sceneref::is_action(t));
            set_char_box(&ui, idx, &new_prompt);
            match idx {
                0 => ui.set_scene_override_c1("".into()),
                1 => ui.set_scene_override_c2("".into()),
                2 => ui.set_scene_override_c3("".into()),
                _ => ui.set_scene_override_c4("".into()),
            }
            let changed = {
                let mut st = state.borrow_mut();
                if let Some(r) = st.scene_ref.as_mut() {
                    sceneref::clear_char_actions(&mut r.actions, idx);
                    true
                } else {
                    false
                }
            };
            if changed {
                apply_scene_ref(&ui, &state);
            }
            sync_scene_panel(&ui, &state);
            analyze(&ui, &state);
            ui.set_scene_ref_status(format!("Character {} actions cleared.", idx + 1).into());
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_char_text_edited(move |which, text| {
            let Some(ui) = weak.upgrade() else { return };
            let idx = which.max(0) as usize;
            let current = char_box(&ui, idx);
            let action_tags: Vec<String> = parser::parse_prompt(&text)
                .into_iter()
                .map(|t| t.raw.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            // exactly the tags in the card: removing one removes it, emptying the card removes all
            let new_prompt = sceneref::replace(&current, &action_tags, &sceneref::is_action);
            set_char_box(&ui, idx, &new_prompt);
            if action_tags.is_empty() {
                // an empty card means "none", not "use the image's": drop them from the reading too
                if let Some(r) = state.borrow_mut().scene_ref.as_mut() {
                    sceneref::clear_char_actions(&mut r.actions, idx);
                }
            }
            match idx {
                0 => ui.set_scene_override_c1(text.clone()),
                1 => ui.set_scene_override_c2(text.clone()),
                2 => ui.set_scene_override_c3(text.clone()),
                _ => ui.set_scene_override_c4(text.clone()),
            }
            analyze(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_action_move_requested(move |from, to| {
            let Some(ui) = weak.upgrade() else { return };
            let from = from.max(0) as usize;
            let to = to.max(0) as usize;
            let changed = {
                let mut st = state.borrow_mut();
                if let Some(r) = st.scene_ref.as_mut() {
                    sceneref::move_char_actions(&mut r.actions, from, to);
                    true
                } else {
                    false
                }
            };
            if changed {
                apply_scene_ref(&ui, &state);
                sync_scene_panel(&ui, &state);
                ui.set_scene_ref_status(format!("Moved pose from Character {} to Character {}.", from + 1, to + 1).into());
            }
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_remove_char_image_requested(move |which| {
            let Some(ui) = weak.upgrade() else { return };
            let idx = (which.max(0) as usize).min(3);
            {
                let mut st = state.borrow_mut();
                st.char_paths[idx] = None;
                st.image_gens.chars[idx] += 1;
                st.char_images[idx] = None;
                st.char_thumbs[idx] = None;
            }
            match idx {
                0 => { ui.set_char1_image_loaded(false); ui.set_char1_image_thumb(slint::Image::default()); }
                1 => { ui.set_char2_image_loaded(false); ui.set_char2_image_thumb(slint::Image::default()); }
                2 => { ui.set_char3_image_loaded(false); ui.set_char3_image_thumb(slint::Image::default()); }
                _ => { ui.set_char4_image_loaded(false); ui.set_char4_image_thumb(slint::Image::default()); }
            }
            update_pending(&ui, &state);
            ui.set_status_message(format!("Character {} reference image removed.", idx + 1).into());
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_clear_char_requested(move |which| {
            let Some(ui) = weak.upgrade() else { return };
            let idx = (which.max(0) as usize).min(3);
            match idx {
                0 => { ui.set_raw_prompt("".into()); ui.set_lora_tags("".into()); }
                1 => { ui.set_char2_prompt("".into()); ui.set_char2_lora_tags("".into()); }
                2 => { ui.set_char3_prompt("".into()); ui.set_char3_lora_tags("".into()); }
                _ => { ui.set_char4_prompt("".into()); ui.set_char4_lora_tags("".into()); }
            }
            analyze(&ui, &state);
            ui.set_status_message(format!("Character {} cleared.", idx + 1).into());
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_identity_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            // re-filter what was already read, no models run again
            image_extract(&ui, &state, false);
            for which in 0..tab_chars(&ui) {
                if state.borrow().char_images[which].is_some() {
                    apply_char_image(&ui, &state, which);
                }
            }
        });
    }
    {
        let weak = ui.as_weak();
        let _state = state.clone();
        ui.on_second_pass_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            save_settings(&ui);
            let on = ui.get_image_second_pass();
            let msg = if on {
                "2nd pass visual check enabled (vision models directly rescan image pixels)."
            } else {
                "2nd pass visual check disabled (pure text synthesis from pass 1 description)."
            };
            ui.set_image_status(msg.into());
            with_state(|state| {
                state.borrow_mut().image_ai = None;
                state.borrow_mut().two_ai = None;
            });
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_body_shape_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            save_settings(&ui);
            let on = effective_scene_body_shape(&ui);
            let msg = if on {
                "Scene Body Shape enabled: Scene supplies body proportions & shape; Character keeps outfit, stature & face."
            } else {
                "Scene Body Shape disabled: Character supplies body proportions & shape."
            };
            ui.set_scene_ref_status(msg.into());
            {
                let mut st = state.borrow_mut();
                if let Some(ref mut sr) = st.scene_ref {
                    if !sr.raw_tags.is_empty() {
                        if sr.raw_tags.contains(" || ") {
                            let parts: Vec<&str> = sr.raw_tags.split(" || ").collect();
                            let char_slices: Vec<String> = parts[0].split(" | ").map(|s| s.to_string()).collect();
                            sr.actions = sceneref::from_split_with_shape(&char_slices, parts.get(1).unwrap_or(&""), false, on);
                        } else {
                            sr.actions = sceneref::from_single_with_shape(&sr.raw_tags, false, on);
                        }
                    }
                }
            }
            for which in 0..tab_chars(&ui) {
                if state.borrow().char_images[which].is_some() {
                    apply_char_image(&ui, &state, which);
                }
            }
            apply_scene_ref(&ui, &state);
            analyze(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_dual_pass_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            save_settings(&ui);
            let on = ui.get_enable_dual_pass();
            let msg = if on {
                "Dual-Pass Vision enabled (Moondream 2-pass + WD14 reconciliation)."
            } else {
                "Dual-Pass Vision disabled (single-pass reading)."
            };
            ui.set_image_status(msg.into());
            let shape_on = effective_scene_body_shape(&ui);
            {
                let mut st = state.borrow_mut();
                st.image_ai = None;
                st.two_ai = None;
                if let Some(ref mut sr) = st.scene_ref {
                    if !sr.raw_tags.is_empty() {
                        if sr.raw_tags.contains(" || ") {
                            let parts: Vec<&str> = sr.raw_tags.split(" || ").collect();
                            let char_slices: Vec<String> = parts[0].split(" | ").map(|s| s.to_string()).collect();
                            sr.actions = sceneref::from_split_with_shape(&char_slices, parts.get(1).unwrap_or(&""), false, shape_on);
                        } else {
                            sr.actions = sceneref::from_single_with_shape(&sr.raw_tags, false, shape_on);
                        }
                    }
                }
            }
            for which in 0..tab_chars(&ui) {
                if state.borrow().char_images[which].is_some() {
                    apply_char_image(&ui, &state, which);
                }
            }
            apply_scene_ref(&ui, &state);
            analyze(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_dual_pass_mode_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            save_settings(&ui);
            let mode_idx = ui.get_dual_pass_mode_index();
            let msg = if mode_idx == 0 {
                "Dual-Pass Mode: Scene + Body Shape (Mode A)"
            } else {
                "Dual-Pass Mode: Scene + Skeletal Pose (Mode B)"
            };
            ui.set_image_status(msg.into());
            let shape_on = effective_scene_body_shape(&ui);
            {
                let mut st = state.borrow_mut();
                st.image_ai = None;
                st.two_ai = None;
                if let Some(ref mut sr) = st.scene_ref {
                    if !sr.raw_tags.is_empty() {
                        if sr.raw_tags.contains(" || ") {
                            let parts: Vec<&str> = sr.raw_tags.split(" || ").collect();
                            let char_slices: Vec<String> = parts[0].split(" | ").map(|s| s.to_string()).collect();
                            sr.actions = sceneref::from_split_with_shape(&char_slices, parts.get(1).unwrap_or(&""), false, shape_on);
                        } else {
                            sr.actions = sceneref::from_single_with_shape(&sr.raw_tags, false, shape_on);
                        }
                    }
                }
            }
            for which in 0..tab_chars(&ui) {
                if state.borrow().char_images[which].is_some() {
                    apply_char_image(&ui, &state, which);
                }
            }
            apply_scene_ref(&ui, &state);
            analyze(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_llm_synthesis_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            save_settings(&ui);
            let on = ui.get_use_llm_synthesis();
            let msg = if on {
                "LLM Tag Synthesis enabled (Gemma 4 12B in Ollama)."
            } else {
                "LLM Tag Synthesis disabled (direct mechanical tag merge)."
            };
            ui.set_image_status(msg.into());
            let mut st = state.borrow_mut();
            st.image_ai = None;
            st.two_ai = None;
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_image_mode_changed(move || {
            if let Some(ui) = weak.upgrade() {
                image_extract(&ui, &state, false);
            }
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_tag_threshold_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            if ui.get_app_tab() >= 1 {
                if !ui.get_image_use_ai() && state.borrow().two_tags.is_some() {
                    two_image_extract(&ui, &state, false);
                }
                return;
            }
            let has_tags = state.borrow().last_tags.is_some();
            if has_tags && !ui.get_image_use_ai() {
                apply_tags(&ui, &state, None);
            }
        });
    }
    {
        let weak = ui.as_weak();
        ui.on_download_tagger_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            ai_engine::reset_cancel();
    ui.set_tagger_busy(true);
            let (tx, rx) = mpsc::channel::<String>();
            let progress_weak = weak.clone();
            std::thread::spawn(move || {
                for msg in rx {
                    let _ = progress_weak.upgrade_in_event_loop(move |ui| ui.set_image_status(msg.into()));
                }
            });
            let done_weak = weak.clone();
            std::thread::spawn(move || {
                let result = tagger::download(tx);
                let _ = done_weak.upgrade_in_event_loop(move |ui| {
                    ui.set_tagger_busy(false);
                    ui.set_tagger_installed(tagger::is_installed());
                    match result {
                        Ok(()) => with_state(|state| {
                            if state.borrow().image_path.is_some() {
                                run_tagger(&ui, state);
                            } else {
                                ui.set_image_status("Danbooru tagger ready. Click From Image… to tag a picture.".into());
                            }
                        }),
                        Err(e) => ui.set_image_status(format!("Tagger download failed: {e:#}").into()),
                    }
                });
            });
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_undo_image_prompt_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let two = state.borrow_mut().two_before_image.take();
            if let Some((boxes, scene)) = two {
                for (i, b) in boxes.iter().enumerate() {
                    set_char_box(&ui, i, b);
                }
                ui.set_scene_prompt(scene.into());
                ui.set_image_can_undo(false);
                analyze(&ui, &state);
                ui.set_status_message("Restored your previous prompts.".into());
                return;
            }
            let previous = state.borrow_mut().prompt_before_image.take();
            if let Some(text) = previous {
                ui.set_raw_prompt(text.into());
                ui.set_image_can_undo(false);
                analyze(&ui, &state);
                ui.set_status_message("Restored your previous prompt.".into());
            }
        });
    }
}
