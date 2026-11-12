//! "Send to ComfyUI" bridge, app side (Gemini's Track 2, 2026-11-11): after every render the prompt the
//! user sees goes to `~/.cache/prompt-condenser/live_prompt.json`, which the ComfyUI nodes read when their
//! `live_stream` input is on. A background writer keeps only the newest state and writes at most once a
//! second, only when it changed, through a temp file + rename so a node never reads half a file.
//! A second thread asks ComfyUI (`127.0.0.1:8188/system_stats`, 300 ms timeout) every 10 s whether it runs,
//! for the status pill. "⚡ Queue" (`wire_queue`) writes the shown prompt at once (`flush`, past the 1 s
//! limit) and then POSTs `127.0.0.1:8188/prompt_condenser/queue` (2 s timeout); the add-on's route
//! tells its JS extension (`tools/comfyui/js/queue.js`) to press Queue in the open ComfyUI tab.

use super::*;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const WRITE_EVERY: Duration = Duration::from_secs(1);
const PROBE_EVERY: Duration = Duration::from_secs(10);
const PROBE_TIMEOUT: Duration = Duration::from_millis(300);
const COMFYUI_URL: &str = "http://127.0.0.1:8188/system_stats";
const QUEUE_URL: &str = "http://127.0.0.1:8188/prompt_condenser/queue";
const QUEUE_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct LiveChange {
    pub(crate) original: String,
    pub(crate) suggestion: String,
    pub(crate) reason: String,
}

/// What the nodes read (field names are the contract with `tools/comfyui/nodes.py`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct LivePrompt {
    /// Seconds since the Unix epoch (the nodes treat a file older than 600 s as stale).
    pub(crate) timestamp: u64,
    pub(crate) positive: String,
    pub(crate) negative: String,
    /// Each character's own block on the 2-4 Characters tabs, else empty.
    pub(crate) characters: Vec<String>,
    pub(crate) view: String,
    pub(crate) checkpoint: String,
    /// The Copy VRP regions JSON on the 2-4 Characters tabs, else "".
    pub(crate) vrp_state: String,
    /// "edited" / "detector" / "strips", or "" without regions.
    pub(crate) regions_source: String,
    pub(crate) changes: Vec<LiveChange>,
    pub(crate) tokens_before: usize,
    pub(crate) tokens_after: usize,
}

impl LivePrompt {
    /// Same content, ignoring the time.
    fn same_as(&self, o: &LivePrompt) -> bool {
        LivePrompt { timestamp: 0, ..self.clone() } == LivePrompt { timestamp: 0, ..o.clone() }
    }
}

pub(crate) fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub(crate) fn live_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/prompt-condenser/live_prompt.json"))
}

/// Writes through a temp file in the same folder, then renames it over the target.
pub(crate) fn write_atomic(path: &Path, live: &LivePrompt) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(live).map_err(std::io::Error::other)?)?;
    std::fs::rename(&tmp, path)
}

/// What the writer thread is sent: a state to write within the 1 s limit, or one to write right away
/// (⚡ Queue), with the write's result sent back.
enum Msg {
    Publish(LivePrompt),
    Flush(LivePrompt, mpsc::Sender<std::io::Result<()>>),
}

thread_local! {
    static SENDER: std::cell::RefCell<Option<mpsc::Sender<Msg>>> = const { std::cell::RefCell::new(None) };
    /// The last state `publish` was given, so ⚡ Queue can write it without a render.
    static LAST: std::cell::RefCell<Option<LivePrompt>> = const { std::cell::RefCell::new(None) };
}

/// Starts the writer thread for `path`; later `publish` / `flush` calls feed it.
fn start_writer(path: PathBuf) -> mpsc::Sender<Msg> {
    let (tx, rx) = mpsc::channel::<Msg>();
    std::thread::spawn(move || {
        let mut written: Option<LivePrompt> = None;
        let mut pending: Option<LivePrompt> = None;
        let mut last_write = Instant::now() - WRITE_EVERY;
        loop {
            let wait = if pending.is_some() { WRITE_EVERY.saturating_sub(last_write.elapsed()) } else { Duration::from_secs(3600) };
            let mut msgs = Vec::new();
            match rx.recv_timeout(wait) {
                Ok(m) => msgs.push(m),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            // drain to the newest state
            while let Ok(m) = rx.try_recv() {
                msgs.push(m);
            }
            for m in msgs {
                match m {
                    Msg::Publish(live) => pending = Some(live),
                    Msg::Flush(live, done) => {
                        // written now; an older pending state must not replace it afterwards
                        pending = None;
                        let r = write_atomic(&path, &live);
                        if r.is_ok() {
                            last_write = Instant::now();
                            written = Some(live);
                        }
                        let _ = done.send(r);
                    }
                }
            }
            if last_write.elapsed() < WRITE_EVERY {
                continue;
            }
            if let Some(live) = pending.take() {
                if written.as_ref().is_some_and(|w| w.same_as(&live)) {
                    continue;
                }
                if write_atomic(&path, &live).is_ok() {
                    last_write = Instant::now();
                    written = Some(live);
                }
            }
        }
    });
    tx
}

fn send(msg: Msg) -> bool {
    SENDER.with(|s| {
        let mut s = s.borrow_mut();
        if s.is_none() {
            *s = live_path().map(start_writer);
        }
        s.as_ref().is_some_and(|tx| tx.send(msg).is_ok())
    })
}

/// Hands the current prompt to the writer (UI thread).
pub(crate) fn publish(live: LivePrompt) {
    LAST.with(|l| *l.borrow_mut() = Some(live.clone()));
    send(Msg::Publish(live));
}

/// Writes `live` right away (no 1 s limit); the receiver gets the write's result (UI thread).
pub(crate) fn flush(live: LivePrompt) -> Option<mpsc::Receiver<std::io::Result<()>>> {
    let (tx, rx) = mpsc::channel();
    send(Msg::Flush(live, tx)).then_some(rx)
}

/// What the add-on's queue route answered.
#[derive(Debug, PartialEq)]
pub(crate) enum QueueReply {
    Queued,
    /// 404: an add-on from before the route.
    TooOld,
    Unreachable,
    Other(u16),
}

impl QueueReply {
    pub(crate) fn message(&self) -> String {
        match self {
            QueueReply::Queued => "✓ Queued in ComfyUI (the open ComfyUI tab runs its workflow with the live prompt)".into(),
            QueueReply::TooOld => "The ComfyUI add-on is too old: ⚡ Install Node, then restart ComfyUI".into(),
            QueueReply::Unreachable => "ComfyUI not reachable".into(),
            QueueReply::Other(code) => format!("ComfyUI answered {code} to the queue request"),
        }
    }
}

/// POSTs the add-on's queue route.
pub(crate) fn request_queue(url: &str, timeout: Duration) -> QueueReply {
    let Ok(client) = reqwest::blocking::Client::builder().timeout(timeout).connect_timeout(timeout).build() else {
        return QueueReply::Unreachable;
    };
    match client.post(url).header("Content-Type", "application/json").body("{}").send() {
        Ok(r) if r.status().is_success() => QueueReply::Queued,
        Ok(r) if r.status().as_u16() == 404 => QueueReply::TooOld,
        Ok(r) => QueueReply::Other(r.status().as_u16()),
        Err(_) => QueueReply::Unreachable,
    }
}

/// "⚡ Queue": writes the prompt shown in After (a 🎲 Roll pick included) to live_prompt.json at once,
/// then asks ComfyUI to queue the open workflow (background thread).
pub(crate) fn wire_queue(ui: &AppWindow) {
    let weak = ui.as_weak();
    ui.on_queue_comfyui_requested(move || {
        let Some(ui) = weak.upgrade() else { return };
        let Some(mut live) = LAST.with(|l| l.borrow().clone()) else {
            ui.set_status_message("Nothing to queue yet: write a prompt first.".into());
            return;
        };
        live.positive = ui.get_condensed_prompt().to_string();
        live.tokens_after = parser::estimate_clip_tokens(&live.positive).0;
        live.timestamp = now_secs();
        let Some(written) = flush(live) else {
            ui.set_status_message("Can't write live_prompt.json (no $HOME).".into());
            return;
        };
        ui.set_status_message("⚡ Queueing in ComfyUI…".into());
        let weak = ui.as_weak();
        std::thread::spawn(move || {
            let msg = match written.recv_timeout(QUEUE_TIMEOUT) {
                Ok(Ok(())) => request_queue(QUEUE_URL, QUEUE_TIMEOUT).message(),
                Ok(Err(e)) => format!("Can't write live_prompt.json: {e}"),
                Err(_) => "Can't write live_prompt.json (the writer didn't answer)".into(),
            };
            let _ = weak.upgrade_in_event_loop(move |ui| ui.set_status_message(msg.into()));
        });
    });
}

/// Whether ComfyUI answers at `url` within the timeout.
pub(crate) fn comfyui_online(url: &str, timeout: Duration) -> bool {
    reqwest::blocking::Client::builder()
        .timeout(timeout)
        .connect_timeout(timeout)
        .build()
        .ok()
        .and_then(|c| c.get(url).send().ok())
        .is_some_and(|r| r.status().is_success())
}

/// Probes ComfyUI every 10 s and shows the result in the status pill.
pub(crate) fn start_probe(ui: &AppWindow) {
    let weak = ui.as_weak();
    std::thread::spawn(move || loop {
        let online = comfyui_online(COMFYUI_URL, PROBE_TIMEOUT);
        if weak.upgrade_in_event_loop(move |ui| ui.set_comfyui_online(online)).is_err() {
            break;
        }
        std::thread::sleep(PROBE_EVERY);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> LivePrompt {
        LivePrompt {
            timestamp: 1_794_428_790,
            positive: "score_9, 1girl, smile".into(),
            negative: "lowres".into(),
            characters: vec![],
            view: "None".into(),
            checkpoint: "PonyV6".into(),
            vrp_state: String::new(),
            regions_source: String::new(),
            changes: vec![LiveChange { original: "masterpiece".into(), suggestion: "score_9".into(), reason: "Pony".into() }],
            tokens_before: 3,
            tokens_after: 5,
        }
    }

    #[test]
    fn live_prompt_has_the_fields_the_nodes_read() {
        let v = serde_json::to_value(sample()).unwrap();
        for key in ["timestamp", "positive", "negative", "characters", "view", "checkpoint", "vrp_state", "regions_source", "changes", "tokens_before", "tokens_after"] {
            assert!(v.get(key).is_some(), "missing {key}");
        }
        assert!(v["timestamp"].is_u64() && v["characters"].is_array() && v["tokens_after"].is_u64());
        assert_eq!(v["changes"][0]["suggestion"], "score_9");
        assert!(v["vrp_state"].is_string());
        // the age check needs seconds, not milliseconds
        assert!(now_secs() < 10_000_000_000);
    }

    #[test]
    fn writes_are_atomic_and_complete() {
        let dir = std::env::temp_dir().join(format!("pc-live-{}", std::process::id()));
        let path = dir.join("sub/live_prompt.json");
        write_atomic(&path, &sample()).unwrap();
        let back: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(back["positive"], "score_9, 1girl, smile");
        assert!(!path.with_extension("json.tmp").exists());
        let mut newer = sample();
        newer.positive = "score_9, 1girl, grin".into();
        write_atomic(&path, &newer).unwrap();
        let back: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(back["positive"], "score_9, 1girl, grin");
        assert!(sample().same_as(&LivePrompt { timestamp: 1, ..sample() }));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Answers every request with `status` (a one-request-per-connection HTTP stub).
    fn stub(status: &'static str) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let _ = write!(stream, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}");
            }
        });
        format!("http://{addr}/prompt_condenser/queue")
    }

    #[test]
    fn queue_replies_map_to_status_messages() {
        let t = Duration::from_secs(2);
        assert_eq!(request_queue(&stub("200 OK"), t), QueueReply::Queued);
        assert_eq!(request_queue(&stub("404 Not Found"), t), QueueReply::TooOld);
        assert_eq!(request_queue(&stub("500 Internal Server Error"), t), QueueReply::Other(500));
        assert_eq!(request_queue("http://127.0.0.1:9/prompt_condenser/queue", Duration::from_millis(300)), QueueReply::Unreachable);
        assert!(QueueReply::Queued.message().starts_with("✓ Queued in ComfyUI"));
        assert!(QueueReply::TooOld.message().contains("Install Node"));
        assert_eq!(QueueReply::Unreachable.message(), "ComfyUI not reachable");
    }

    #[test]
    fn a_flush_is_written_at_once_and_beats_an_older_pending_state() {
        let dir = std::env::temp_dir().join(format!("pc-live-flush-{}", std::process::id()));
        let path = dir.join("live_prompt.json");
        let tx = start_writer(path.clone());
        tx.send(Msg::Publish(sample())).unwrap();
        // a publish right after the first write waits for the 1 s limit; the flush doesn't
        let mut older = sample();
        older.positive = "older".into();
        tx.send(Msg::Publish(older)).unwrap();
        let mut rolled = sample();
        rolled.positive = "score_9, 1girl, grin".into();
        let (done_tx, done) = mpsc::channel();
        tx.send(Msg::Flush(rolled, done_tx)).unwrap();
        done.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
        let read = || serde_json::from_slice::<serde_json::Value>(&std::fs::read(&path).unwrap()).unwrap()["positive"].clone();
        assert_eq!(read(), "score_9, 1girl, grin");
        std::thread::sleep(Duration::from_millis(1300));
        assert_eq!(read(), "score_9, 1girl, grin", "the older pending state overwrote the flush");
        drop(tx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_closed_port_reads_as_offline() {
        // port 9 (discard) is closed on a normal desktop
        assert!(!comfyui_online("http://127.0.0.1:9/system_stats", Duration::from_millis(300)));
    }
}
