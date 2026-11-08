//! Prompt-Condenser: dual-pass Moondream + WD14 pipeline (Rust port of
//! `prompt_condenser_dual_pass.py`, which stays the reference implementation).
//!
//! Uses only the standard library, so it drops into any crate: add this file as
//! `src/prompt_condenser.rs` and declare it in main.rs / lib.rs:
//!
//! ```text
//! #[allow(dead_code)] // until every helper is wired up
//! mod prompt_condenser;
//! #[cfg(test)]
//! mod prompt_condenser_parity; // optional: checks Rust == the Python reference
//! ```
//!
//! Flow (same as the Python reference):
//!   1. WD14 tags the image; Moondream describes it in two passes (prose).
//!   2. Moondream prose is converted to WD-style tags.
//!   3. WD14 adjusts the Moondream tags: WD14 wins on attributes (colors, hair
//!      length, breast size), Moondream wins on pose and camera angle.
//!   4. The merged tags are split into scene / character prompts by mode.
//!   5. Optionally Gemma (Ollama) polishes the result, optionally with the image.
//!
//! All I/O is injected: you pass closures for Moondream and Gemma, so this module
//! has no HTTP, image or async dependencies. Every failure falls back to the
//! mechanical merge and is reported in `PipelineResult::warnings`.
//!
//! The single-pass path is not here: keep your existing code for it and call
//! `run_dual_pass` only when the dual-pass toggle is on.

use std::collections::{BTreeSet, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Constants (Phase 0)
// ---------------------------------------------------------------------------
pub const VLM_TEMPERATURE: f32 = 0.1;
pub const MOONDREAM_TEMPERATURE: f32 = VLM_TEMPERATURE;
pub const GEMMA_TEMPERATURE: f32 = 0.35;
/// Pass these to your WD14 tagger; this module only receives the tag string.
pub const WD14_GENERAL_THRESHOLD: f32 = 0.22;
pub const WD14_CHARACTER_THRESHOLD: f32 = 0.85;
/// 128K, the context the model page lists for the 12b tag. Input is only ~1.5K
/// tokens, so lower this (8192 or 16384) if the GPU runs short of memory.
pub const GEMMA_CONTEXT_WINDOW: u32 = 131_072;
pub const GEMMA_MAX_TOKENS: u32 = 768;
pub const GEMMA_MAX_TOKENS_RETRY: u32 = 1024;
pub const VLM_RETRIES: u32 = 1;
pub const MOONDREAM_RETRIES: u32 = VLM_RETRIES;
pub const VLM_RETRY_TEMP_BUMP: f32 = 0.2;
pub const MOONDREAM_RETRY_TEMP_BUMP: f32 = VLM_RETRY_TEMP_BUMP;

/// Runtime settings; `Config::default()` matches the Python constants.
#[derive(Clone, Debug)]
pub struct Config {
    /// Prepended to the scene prompt; empty string disables it.
    pub quality_prefix: String,
    /// Write "(" as "\(" (A1111/ComfyUI prompt syntax).
    pub escape_parentheses: bool,
    /// Mechanical path only: also append VLM wording that matched no tag.
    pub keep_unmatched_prose: bool,
    pub vlm_temperature: f32,
    pub vlm_retries: u32,
    pub vlm_retry_temp_bump: f32,
    pub moondream_temperature: f32,
    pub moondream_retries: u32,
    pub moondream_retry_temp_bump: f32,
    pub gemma_temperature: f32,
    pub gemma_max_tokens: u32,
    pub gemma_max_tokens_retry: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            quality_prefix: QUALITY_PREFIX_DEFAULT.to_string(),
            escape_parentheses: false,
            keep_unmatched_prose: false,
            vlm_temperature: VLM_TEMPERATURE,
            vlm_retries: VLM_RETRIES,
            vlm_retry_temp_bump: VLM_RETRY_TEMP_BUMP,
            moondream_temperature: MOONDREAM_TEMPERATURE,
            moondream_retries: MOONDREAM_RETRIES,
            moondream_retry_temp_bump: MOONDREAM_RETRY_TEMP_BUMP,
            gemma_temperature: GEMMA_TEMPERATURE,
            gemma_max_tokens: GEMMA_MAX_TOKENS,
            gemma_max_tokens_retry: GEMMA_MAX_TOKENS_RETRY,
        }
    }
}

