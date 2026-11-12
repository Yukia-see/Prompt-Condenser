//! Prompt history: the 15 most recent prompt states plus up to 10 starred ones (never evicted),
//! captured after a 2.5 s typing pause and on "⭐ Stash", saved to
//! `~/.config/prompt-condenser/history.json` on exit. Restore puts a snapshot back into the boxes
//! after saving the current state, so a restore can itself be undone from the list.

use super::*;
use serde::{Deserialize, Serialize};

/// Unstarred snapshots kept (oldest dropped first).
pub(crate) const MAX_RECENT: usize = 15;
/// Starred snapshots allowed; they are never dropped.
pub(crate) const MAX_STARRED: usize = 10;
/// Typing pause before the boxes are captured.
const CAPTURE_PAUSE: std::time::Duration = std::time::Duration::from_millis(2500);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub(crate) struct PromptSnapshot {
    /// Seconds since the Unix epoch.
    pub(crate) timestamp: u64,
    /// The first ~35 characters, for the list.
    pub(crate) label: String,
    /// Before (character 1).
    pub(crate) raw_prompt: String,
    /// Characters 2-4 (their boxes, even when the tab hides them).
    pub(crate) char_prompts: Vec<String>,
    pub(crate) scene_prompt: String,
    /// How many characters the tab showed (1-4).
    pub(crate) char_count: i32,
    pub(crate) starred: bool,
}

impl PromptSnapshot {
    /// Same text in every box (time, label and star don't count).
    fn same_content(&self, o: &PromptSnapshot) -> bool {
        self.raw_prompt == o.raw_prompt && self.char_prompts == o.char_prompts && self.scene_prompt == o.scene_prompt
    }

    fn is_empty(&self) -> bool {
        self.raw_prompt.trim().is_empty()
            && self.scene_prompt.trim().is_empty()
            && self.char_prompts.iter().all(|c| c.trim().is_empty())
    }
}

/// The first ~35 characters of the first non-empty box, on one line.
pub(crate) fn label_of(texts: &[&str]) -> String {
    let text = texts.iter().map(|t| t.trim()).find(|t| !t.is_empty()).unwrap_or("");
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > 35 {
        format!("{}…", flat.chars().take(35).collect::<String>().trim_end())
    } else {
        flat
    }
}

/// Newest first.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct History {
    pub(crate) items: Vec<PromptSnapshot>,
}

impl History {
    /// Adds a snapshot at the top unless it is empty or the newest one has the same text.
    /// Returns whether it was added.
    pub(crate) fn push(&mut self, snap: PromptSnapshot) -> bool {
        if snap.is_empty() || self.items.first().is_some_and(|s| s.same_content(&snap)) {
            return false;
        }
        self.items.insert(0, snap);
        self.trim();
        true
    }

    /// Keeps every starred snapshot and the newest `MAX_RECENT` others.
    fn trim(&mut self) {
        let mut recent = 0;
        self.items.retain(|s| {
            if s.starred {
                return true;
            }
            recent += 1;
            recent <= MAX_RECENT
        });
    }

    /// Stars / unstars one; starring an 11th is refused (returns false).
    pub(crate) fn toggle_star(&mut self, index: usize) -> bool {
        let starred = self.items.iter().filter(|s| s.starred).count();
        let Some(s) = self.items.get_mut(index) else { return false };
        if !s.starred && starred >= MAX_STARRED {
            return false;
        }
        s.starred = !s.starred;
        self.trim();
        true
    }

    fn path() -> Option<std::path::PathBuf> {
        settings_path().map(|p| p.with_file_name("history.json"))
    }

    pub(crate) fn load() -> History {
        Self::path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|t| serde_json::from_str::<History>(&t).ok())
            .map(|mut h| {
                h.trim();
                h
            })
            .unwrap_or_default()
    }

    pub(crate) fn save(&self) {
        if let Some(path) = Self::path() {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Ok(json) = serde_json::to_string_pretty(self) {
                let _ = std::fs::write(path, json);
            }
        }
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// "just now", "5 min ago", "3 h ago", "2 d ago".
pub(crate) fn relative_time(then: u64, now: u64) -> String {
    let s = now.saturating_sub(then);
    match s {
        0..=59 => "just now".to_string(),
        60..=3599 => format!("{} min ago", s / 60),
        3600..=86_399 => format!("{} h ago", s / 3600),
        _ => format!("{} d ago", s / 86_400),
    }
}

/// The boxes as they are now.
pub(crate) fn current(ui: &AppWindow) -> PromptSnapshot {
    let raw = ui.get_raw_prompt().to_string();
    let chars = vec![ui.get_char2_prompt().to_string(), ui.get_char3_prompt().to_string(), ui.get_char4_prompt().to_string()];
    let scene = ui.get_scene_prompt().to_string();
    PromptSnapshot {
        timestamp: now(),
        label: label_of(&[&raw, &chars[0], &chars[1], &chars[2], &scene]),
        raw_prompt: raw,
        char_prompts: chars,
        scene_prompt: scene,
        char_count: ui.get_app_tab() + 1,
        starred: false,
    }
}

/// Fills the overlay list and the pill's count.
pub(crate) fn push_rows(ui: &AppWindow, h: &History) {
    let t = now();
    let rows: Vec<HistoryRow> = h
        .items
        .iter()
        .map(|s| HistoryRow {
            label: if s.label.is_empty() { "(empty)".into() } else { s.label.clone().into() },
            when: relative_time(s.timestamp, t).into(),
            starred: s.starred,
            chars: s.char_count,
        })
        .collect();
    ui.set_history_count(rows.len() as i32);
    ui.set_history_rows(ModelRc::new(VecModel::from(rows)));
}

thread_local! {
    static CAPTURE_TIMER: slint::Timer = slint::Timer::default();
}

/// Called after every analysis: captures the boxes once typing has paused for 2.5 s.
pub(crate) fn touch(ui: &AppWindow, state: &State) {
    let weak = ui.as_weak();
    let state = state.clone();
    CAPTURE_TIMER.with(|t| {
        t.start(slint::TimerMode::SingleShot, CAPTURE_PAUSE, move || {
            let Some(ui) = weak.upgrade() else { return };
            let mut st = state.borrow_mut();
            if st.history.push(current(&ui)) {
                push_rows(&ui, &st.history);
            }
        })
    });
}

/// Puts a snapshot back into the boxes.
fn restore(ui: &AppWindow, s: &PromptSnapshot) {
    ui.set_raw_prompt(s.raw_prompt.clone().into());
    let get = |i: usize| s.char_prompts.get(i).cloned().unwrap_or_default();
    ui.set_char2_prompt(get(0).into());
    ui.set_char3_prompt(get(1).into());
    ui.set_char4_prompt(get(2).into());
    ui.set_scene_prompt(s.scene_prompt.clone().into());
    ui.set_app_tab((s.char_count - 1).clamp(0, 3));
}

/// Wires the History overlay: Stash, star, Restore.
pub(crate) fn wire_history(ui: &AppWindow, state: &State) {
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_history_open_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            push_rows(&ui, &state.borrow().history);
            ui.set_history_open(true);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_history_stash_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let mut st = state.borrow_mut();
            let mut snap = current(&ui);
            snap.starred = st.history.items.iter().filter(|s| s.starred).count() < MAX_STARRED;
            // the same text as the newest: star that one instead of adding a copy
            let msg = if st.history.items.first().is_some_and(|s| s.same_content(&snap)) {
                if !st.history.items[0].starred && !st.history.toggle_star(0) {
                    "Stash full: unstar one first (10 starred at most).".to_string()
                } else {
                    "Stashed (starred).".to_string()
                }
            } else if snap.is_empty() {
                "Nothing to stash: the boxes are empty.".to_string()
            } else {
                let starred = snap.starred;
                st.history.push(snap);
                if starred { "Stashed (starred)." } else { "Stashed; 10 are starred already, so it isn't starred." }.to_string()
            };
            push_rows(&ui, &st.history);
            ui.set_status_message(msg.into());
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_history_star_toggled(move |i| {
            let Some(ui) = weak.upgrade() else { return };
            let mut st = state.borrow_mut();
            if !st.history.toggle_star(i as usize) {
                ui.set_status_message("10 snapshots are starred: unstar one first.".into());
            }
            push_rows(&ui, &st.history);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_history_restore_requested(move |i| {
            let Some(ui) = weak.upgrade() else { return };
            let target = {
                let mut st = state.borrow_mut();
                let Some(target) = st.history.items.get(i as usize).cloned() else { return };
                // the current boxes go into the list first, so the restore can be undone
                st.history.push(current(&ui));
                push_rows(&ui, &st.history);
                target
            };
            restore(&ui, &target);
            ui.set_history_open(false);
            analyze(&ui, &state);
            ui.set_status_message(format!("Restored \"{}\"; your previous text is at the top of History.", target.label).into());
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(text: &str, starred: bool) -> PromptSnapshot {
        PromptSnapshot { raw_prompt: text.into(), label: label_of(&[text]), starred, char_count: 1, ..Default::default() }
    }

    #[test]
    fn history_keeps_15_recent_and_every_starred() {
        let mut h = History::default();
        assert!(h.push(snap("a", false)));
        assert!(!h.push(snap("a", false)), "same text as the newest");
        assert!(!h.push(snap("  ", false)), "empty");
        h.items[0].starred = true;
        for i in 0..20 {
            h.push(snap(&format!("p{i}"), false));
        }
        assert_eq!(h.items.len(), MAX_RECENT + 1);
        assert!(h.items.iter().any(|s| s.raw_prompt == "a" && s.starred), "the starred one survives");
        assert_eq!(h.items[0].raw_prompt, "p19");
        // at most 10 starred
        for i in 0..MAX_STARRED - 1 {
            assert!(h.toggle_star(i));
        }
        assert_eq!(h.items.iter().filter(|s| s.starred).count(), MAX_STARRED);
        let unstarred = h.items.iter().position(|s| !s.starred).unwrap();
        assert!(!h.toggle_star(unstarred), "an 11th star is refused");
        // unstarring makes it evictable again
        assert!(h.toggle_star(0));
        assert!(!h.items[0].starred);
    }

    #[test]
    fn history_round_trips_through_json_and_labels_are_short() {
        let mut h = History::default();
        h.push(PromptSnapshot { char_prompts: vec!["1boy".into(), "".into(), "".into()], char_count: 2, ..snap("1girl, beach", true) });
        let back: History = serde_json::from_str(&serde_json::to_string(&h).unwrap()).unwrap();
        assert_eq!(back.items, h.items);
        // an older file without some fields still loads
        let old: History = serde_json::from_str(r#"{"items":[{"raw_prompt":"x"}]}"#).unwrap();
        assert_eq!(old.items[0].raw_prompt, "x");
        assert_eq!(label_of(&["", "  masterpiece,   1girl, very long hair, standing on a beach at sunset"]), "masterpiece, 1girl, very long hair,…");
        assert_eq!(relative_time(100, 100), "just now");
        assert_eq!(relative_time(0, 7200), "2 h ago");
    }
}
