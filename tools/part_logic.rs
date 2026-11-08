
// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Pass 1 = Scene + Body Shape, Pass 2 = Face + Clothes.
    SceneShape,
    /// Pass 1 = Scene + Skeletal Pose, Pass 2 = Character + Body Shape.
    ScenePose,
}

impl Mode {
    pub const ALL: [Mode; 2] = [Mode::SceneShape, Mode::ScenePose];

    /// For the dropdown: "scene_shape" or "scene_pose". Unknown values are an error.
    pub fn parse(s: &str) -> Result<Mode, String> {
        match s {
            "scene_shape" => Ok(Mode::SceneShape),
            "scene_pose" => Ok(Mode::ScenePose),
            other => Err(format!(
                "Unknown dual_pass_mode '{}'; expected one of ('scene_shape', 'scene_pose')",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Mode::SceneShape => "scene_shape",
            Mode::ScenePose => "scene_pose",
        }
    }

    pub fn pass1_prompt(self) -> &'static str {
        match self {
            Mode::SceneShape => MODE_A_PASS1_PROMPT,
            Mode::ScenePose => MODE_B_PASS1_PROMPT,
        }
    }

    pub fn pass2_prompt(self) -> &'static str {
        match self {
            Mode::SceneShape => MODE_A_PASS2_PROMPT,
            Mode::ScenePose => MODE_B_PASS2_PROMPT,
        }
    }

    pub fn llm_instruction(self) -> &'static str {
        match self {
            Mode::SceneShape => MODE_A_LLM_INSTRUCTION,
            Mode::ScenePose => MODE_B_LLM_INSTRUCTION,
        }
    }

    pub fn pass1_keywords(self) -> &'static [&'static str] {
        match self {
            Mode::SceneShape => MODE_A_PASS1_KEYWORDS,
            Mode::ScenePose => MODE_B_PASS1_KEYWORDS,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared text helpers
// ---------------------------------------------------------------------------
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Same as the regex `\bphrase\b` for a phrase that starts and ends with a word char.
pub fn contains_whole_phrase(hay: &str, phrase: &str) -> bool {
    if phrase.is_empty() {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = hay[start..].find(phrase) {
        let abs = start + pos;
        let end = abs + phrase.len();
        let before_ok = match hay[..abs].chars().next_back() {
            Some(c) => !is_word_char(c),
            None => true,
        };
        let after_ok = match hay[end..].chars().next() {
            Some(c) => !is_word_char(c),
            None => true,
        };
        if before_ok && after_ok {
            return true;
        }
        start = abs + hay[abs..].chars().next().map_or(1, |c| c.len_utf8());
    }
    false
}

pub fn norm_tag(tag: &str) -> String {
    tag.trim().to_lowercase().replace('_', " ")
}

/// Join comma-separated chunks, drop empties, de-duplicate (case-insensitive), keep order.
pub fn clean_csv(parts: &[&str]) -> String {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for part in parts {
        for item in part.split(',') {
            let item = item.trim().trim_matches('.').trim();
            if item.is_empty() {
                continue;
            }
            if seen.insert(item.to_lowercase()) {
                out.push(item.to_string());
            }
        }
    }
    out.join(", ")
}

fn is_header_sep(c: char) -> bool {
    matches!(c, ':' | '-' | '\u{2013}' | '\u{2014}')
}

/// Line-level equivalent of `^\s*\d+[.)]\s*\**[^:\n\-–—]+\**\s*[:\-–—]\s*`.
/// Like the Python version, a hyphen inside a zone name also ends the header.
fn strip_header_line(line: &str) -> &str {
    let s = line.trim_start();
    let digits = s.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 {
        return line;
    }
    let rest = &s[digits..];
    if !(rest.starts_with('.') || rest.starts_with(')')) {
        return line;
    }
    let after = &rest[1..];
    match after.char_indices().find(|&(_, c)| is_header_sep(c)) {
        Some((p, c)) if p > 0 => after[p + c.len_utf8()..].trim_start(),
        _ => line,
    }
}

fn strip_zone_headers(text: &str) -> String {
    text.split('\n')
        .map(strip_header_line)
        .collect::<Vec<_>>()
        .join("\n")
        .replace("**", "")
}

pub fn flatten_zones(text: &str) -> String {
    clean_csv(&[&strip_zone_headers(text).replace('\n', ", ")])
}

/// `1girl`, `2boys`, `6+girls`, `solo`, `multiple girls`, `multiple boys`.
pub fn is_subject_tag(tag: &str) -> bool {
    if matches!(tag, "solo" | "multiple girls" | "multiple boys") {
        return true;
    }
    let digits = tag.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 {
        return false;
    }
    let mut rest = &tag[digits..];
    if let Some(r) = rest.strip_prefix('+') {
        rest = r;
    }
    matches!(rest, "girl" | "boy" | "girls" | "boys" | "other" | "others")
}

/// Final output styling: optional escaping of parentheses (not double-escaped).
pub fn style_tags(s: &str, escape_parentheses: bool) -> String {
    if !escape_parentheses {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 4);
    let mut prev: Option<char> = None;
    for c in s.chars() {
        if (c == '(' || c == ')') && prev != Some('\\') {
            out.push('\\');
        }
        out.push(c);
        prev = Some(c);
    }
    out
}

// ---------------------------------------------------------------------------
// Tag vocabulary (WD14 selected_tags.csv)
// ---------------------------------------------------------------------------
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, in_quotes) {
            ('"', true) if chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            ('"', _) => in_quotes = !in_quotes,
            (',', false) => fields.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    fields.push(cur);
    fields
}

/// Tag names from a WD14 `selected_tags.csv` (the `name` column), normalized.
pub fn load_vocab_from_csv(text: &str) -> HashSet<String> {
    let mut vocab = HashSet::new();
    let mut lines = text.lines();
    let header = match lines.next() {
        Some(h) => parse_csv_line(h.trim_start_matches('\u{feff}')),
        None => return vocab,
    };
    let idx = match header.iter().position(|h| h.trim() == "name") {
        Some(i) => i,
        None => return vocab,
    };
    for line in lines {
        if let Some(name) = parse_csv_line(line).get(idx) {
            let n = norm_tag(name);
            if !n.is_empty() {
                vocab.insert(n);
            }
        }
    }
    vocab
}

// ---------------------------------------------------------------------------
// Prose -> WD-style tags
// ---------------------------------------------------------------------------
/// Words and hyphenated words: `[a-z0-9']+(?:-[a-z0-9']+)*` on lowercased text.
fn tokens(clause: &str) -> Vec<String> {
    let lower = clause.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let is_w = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '\'';
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !is_w(chars[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && is_w(chars[i]) {
            i += 1;
        }
        while i + 1 < chars.len() && chars[i] == '-' && is_w(chars[i + 1]) {
            i += 1;
            while i < chars.len() && is_w(chars[i]) {
                i += 1;
            }
        }
        out.push(chars[start..i].iter().collect());
    }
    out
}

fn variants(toks: &[String]) -> Vec<String> {
    let mut out = vec![toks.join(" ")];
    if let Some(last) = toks.last() {
        if last.chars().count() > 3 && last.ends_with('s') {
            let mut v: Vec<String> = toks[..toks.len() - 1].to_vec();
            v.push(last[..last.len() - 1].to_string());
            out.push(v.join(" "));
        }
    }
    out
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProseAnalysis {
    /// Tags found (negated ones excluded), in order.
    pub tags: Vec<String>,
    /// Stated plainly.
    pub strong: BTreeSet<String>,
    /// Stated only with a hedge ("appears to be", "possibly").
    pub weak: BTreeSet<String>,
    /// Explicitly denied ("no glasses", "not wearing boots").
    pub negated: Vec<String>,
}

/// Greedy longest-match of known tags inside Moondream prose, with negation and
/// hedge handling. `is_known` says whether a candidate is in the tag vocabulary.
pub fn analyze_prose(text: &str, is_known: &dyn Fn(&str) -> bool) -> ProseAnalysis {
    const MAX_N: usize = 4;
    let mut res = ProseAnalysis::default();
    let flat = strip_zone_headers(text).replace('\n', ", ");
    for clause in flat.split([',', '.', ';', ':']) {
        let toks = tokens(clause);
        let (mut neg, mut hedge) = (false, false);
        let mut i = 0;
        while i < toks.len() {
            let mut hit: Option<(String, usize)> = None;
            let max = MAX_N.min(toks.len() - i);
            'search: for n in (1..=max).rev() {
                for cand in variants(&toks[i..i + n]) {
                    if is_known(&cand) && !(n == 1 && SINGLE_STOP.contains(&cand.as_str())) {
                        hit = Some((cand, n));
                        break 'search;
                    }
                }
            }
            if let Some((tag, n)) = hit {
                if neg {
                    if !res.negated.contains(&tag) {
                        res.negated.push(tag);
                    }
                } else {
                    if !res.tags.contains(&tag) {
                        res.tags.push(tag.clone());
                    }
                    if hedge {
                        if !res.strong.contains(&tag) {
                            res.weak.insert(tag);
                        }
                    } else {
                        res.weak.remove(&tag);
                        res.strong.insert(tag);
                    }
                }
                i += n;
                continue;
            }
            let tok = toks[i].as_str();
            if NEGATORS.contains(&tok) {
                neg = true;
            } else if HEDGES.contains(&tok) {
                hedge = true;
            } else if SCOPE_END.contains(&tok)
                || (NEW_STATEMENT_VERBS.contains(&tok) && i > 0 && toks[i - 1] == "and")
            {
                neg = false;
                hedge = false;
            }
            i += 1;
        }
    }
    res
}

/// Convenience wrapper: known tags come from a plain set.
pub fn analyze_prose_with_vocab(text: &str, vocab: &HashSet<String>) -> ProseAnalysis {
    analyze_prose(text, &|c| vocab.contains(c))
}

// ---------------------------------------------------------------------------
// Reconcile: WD14 adjusts Moondream
// ---------------------------------------------------------------------------
fn split_color(tag: &str) -> Option<(&str, &str)> {
    let (first, rest) = tag.split_once(' ')?;
    if COLORS.contains(&first) {
        Some((first, rest))
    } else {
        None
    }
}

/// Returns (drop_from_moondream, drop_from_wd14).
pub fn reconcile(md: &[String], wd: &[String]) -> (HashSet<String>, HashSet<String>) {
    let md_set: HashSet<&str> = md.iter().map(|s| s.as_str()).collect();
    let wd_set: HashSet<&str> = wd.iter().map(|s| s.as_str()).collect();
    let mut drop_md: HashSet<String> = HashSet::new();
    let mut drop_wd: HashSet<String> = HashSet::new();

    let mut wd_colors: HashMap<&str, HashSet<&str>> = HashMap::new();
    for t in &wd_set {
        if let Some((color, rest)) = split_color(t) {
            wd_colors.entry(rest).or_default().insert(color);
        }
    }
    for t in &md_set {
        match split_color(t) {
            Some((color, rest)) => {
                if let Some(colors) = wd_colors.get(rest) {
                    if !colors.contains(color) {
                        drop_md.insert(t.to_string()); // WD14 wins on colors
                    }
                }
            }
            None => {
                if wd_colors.contains_key(t) {
                    drop_md.insert(t.to_string()); // plain "dress" yields to "blue dress"
                }
            }
        }
    }

    for group in EXCLUSIVE_GROUPS {
        if group.iter().any(|g| md_set.contains(g)) {
            for g in group.iter() {
                if wd_set.contains(g) && !md_set.contains(g) {
                    drop_wd.insert(g.to_string()); // Moondream wins on pose/camera
                }
            }
        }
    }
    for group in ATTRIBUTE_GROUPS {
        if group.iter().any(|g| wd_set.contains(g)) {
            for g in group.iter() {
                if md_set.contains(g) && !wd_set.contains(g) {
                    drop_md.insert(g.to_string()); // WD14 wins on attribute sizes
                }
            }
        }
    }
    (drop_md, drop_wd)
}

// ---------------------------------------------------------------------------
// Tag router (WD14 tags -> Pass 1 / Pass 2)
// ---------------------------------------------------------------------------
pub fn normalized_tags(raw_wd14_tags: &str) -> Vec<String> {
    raw_wd14_tags
        .split(',')
        .map(norm_tag)
        .filter(|t| !t.is_empty())
        .collect()
}

pub fn sort_tag_lists(raw_wd14_tags: &str, mode: Mode) -> (Vec<String>, Vec<String>) {
    let (mut p1, mut p2) = (Vec::new(), Vec::new());
    for tag in normalized_tags(raw_wd14_tags) {
        if mode.pass1_keywords().iter().any(|k| contains_whole_phrase(&tag, k)) {
            p1.push(tag);
        } else {
            p2.push(tag);
        }
    }
    (p1, p2)
}

// ---------------------------------------------------------------------------
// Merge: convert prose to tags, then let WD14 adjust them
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Merged {
    pub pass1_tags: Vec<String>,
    pub pass2_tags: Vec<String>,
    pub subject_tags: Vec<String>,
    pub dropped_vlm: Vec<String>,
    pub dropped_wd14: Vec<String>,
    pub dropped_hedged: Vec<String>,
    pub dropped_unconfirmed: Vec<String>,
    pub negated_vlm: Vec<String>,
    /// WD14 tags that matched no scene keyword and defaulted to Pass 2; grow the
    /// keyword lists from these.
    pub unrouted_wd14: Vec<String>,
    pub dropped_moondream: Vec<String>,
    pub negated_moondream: Vec<String>,
}

pub fn merge_tags(
    raw_wd14_tags: &str,
    pass1_txt: &str,
    pass2_txt: &str,
    mode: Mode,
    vocab: Option<&HashSet<String>>,
) -> Merged {
    merge_tags_with_options(raw_wd14_tags, pass1_txt, pass2_txt, mode, vocab, false)
}

pub fn merge_tags_with_options(
    raw_wd14_tags: &str,
    pass1_txt: &str,
    pass2_txt: &str,
    mode: Mode,
    vocab: Option<&HashSet<String>>,
    require_two_sources: bool,
) -> Merged {
    let wd_all = normalized_tags(raw_wd14_tags);
    let (wd1, wd2) = sort_tag_lists(raw_wd14_tags, mode);
    let wd_set: HashSet<String> = wd_all.iter().cloned().collect();
    let builtin: HashSet<&str> = BUILTIN_VOCAB.iter().copied().collect();
    let is_known = |c: &str| {
        builtin.contains(c) || wd_set.contains(c) || vocab.is_some_and(|v| v.contains(c))
    };

    let a1 = analyze_prose(pass1_txt, &is_known);
    let a2 = analyze_prose(pass2_txt, &is_known);

    // Hedged VLM tags count only if WD14 agrees.
    let weak: BTreeSet<&String> = a1.weak.iter().chain(a2.weak.iter()).collect();
    let strong: BTreeSet<&String> = a1.strong.iter().chain(a2.strong.iter()).collect();
    let hedged_unconfirmed: BTreeSet<String> = weak
        .difference(&strong)
        .filter(|t| !wd_set.contains(t.as_str()))
        .map(|t| (*t).clone())
        .collect();

    let mut md1: Vec<String> = a1.tags.iter().filter(|t| !hedged_unconfirmed.contains(*t)).cloned().collect();
    let mut md2: Vec<String> = a2.tags.iter().filter(|t| !hedged_unconfirmed.contains(*t)).cloned().collect();
    let mut negated: Vec<String> = a1
        .negated
        .iter()
        .chain(a2.negated.iter())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    negated.sort();

    // Phase 3: Two-Source Rule
    let mut exclusive_all = HashSet::new();
    for group in EXCLUSIVE_GROUPS {
        for item in *group {
            exclusive_all.insert(*item);
        }
    }
    let mut unconfirmed = BTreeSet::new();
    let md1_set: HashSet<String> = md1.iter().cloned().collect();
    let md2_set: HashSet<String> = md2.iter().cloned().collect();

    for t in md1.iter().chain(md2.iter()) {
        if !wd_set.contains(t) && !exclusive_all.contains(t.as_str()) {
            let in_both = md1_set.contains(t) && md2_set.contains(t);
            if !in_both {
                unconfirmed.insert(t.clone());
            }
        }
    }
    if require_two_sources {
        md1.retain(|t| !unconfirmed.contains(t));
        md2.retain(|t| !unconfirmed.contains(t));
    }

    let all_md: Vec<String> = md1.iter().chain(md2.iter()).cloned().collect();
    let all_wd: Vec<String> = wd1.iter().chain(wd2.iter()).cloned().collect();
    let (drop_md, drop_wd) = reconcile(&all_md, &all_wd);

    let subject: Vec<String> = wd_all.iter().filter(|t| is_subject_tag(t)).cloned().collect();
    let keep_wd = |t: &&String| !drop_wd.contains(*t) && !is_subject_tag(t);
    let wd1: Vec<String> = wd1.iter().filter(keep_wd).cloned().collect();
    let wd2: Vec<String> = wd2.iter().filter(keep_wd).cloned().collect();
    let md1: Vec<String> = md1.into_iter().filter(|t| !drop_md.contains(t)).collect();
    let md2: Vec<String> = md2.into_iter().filter(|t| !drop_md.contains(t)).collect();

    let sorted = |s: &HashSet<String>| {
        let mut v: Vec<String> = s.iter().cloned().collect();
        v.sort();
        v
    };
    let drop_vlm_sorted = sorted(&drop_md);
    Merged {
        pass1_tags: md1.into_iter().chain(wd1).collect(),
        pass2_tags: md2.into_iter().chain(wd2.iter().cloned()).collect(),
        subject_tags: subject,
        dropped_vlm: drop_vlm_sorted.clone(),
        dropped_wd14: sorted(&drop_wd),
        dropped_hedged: hedged_unconfirmed.into_iter().collect(),
        dropped_unconfirmed: unconfirmed.into_iter().collect(),
        negated_vlm: negated.clone(),
        unrouted_wd14: wd2,
        dropped_moondream: drop_vlm_sorted,
        negated_moondream: negated,
    }
}

// ---------------------------------------------------------------------------
// Mechanical (no-LLM) path
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prompts {
    pub scene_prompt: String,
    pub character_prompt: String,
}

pub fn process_direct(
    raw_wd14_tags: &str,
    pass1_txt: &str,
    pass2_txt: &str,
    mode: Mode,
    vocab: Option<&HashSet<String>>,
    cfg: &Config,
) -> Prompts {
    let m = merge_tags_with_options(raw_wd14_tags, pass1_txt, pass2_txt, mode, vocab, cfg.require_two_sources);
    let (extra1, extra2) = if cfg.keep_unmatched_prose {
        (flatten_zones(pass1_txt), flatten_zones(pass2_txt))
    } else {
        (String::new(), String::new())
    };
    let scene = clean_csv(&[&cfg.quality_prefix, &m.pass1_tags.join(", "), &extra1]);
    let character = clean_csv(&[&m.subject_tags.join(", "), &m.pass2_tags.join(", "), &extra2]);
    Prompts {
        scene_prompt: style_tags(&scene, cfg.escape_parentheses),
        character_prompt: style_tags(&character, cfg.escape_parentheses),
    }
}

// ---------------------------------------------------------------------------
// Minimal JSON (so the module needs no serde)
// ---------------------------------------------------------------------------
pub mod json {
    #[derive(Clone, Debug, PartialEq)]
    pub enum Value {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<Value>),
        Obj(Vec<(String, Value)>),
    }

    impl Value {
        /// Last matching key wins, like Python's json.loads.
        pub fn get(&self, key: &str) -> Option<&Value> {
            match self {
                Value::Obj(items) => items.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }
    }

    pub fn escape(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    struct Parser {
        chars: Vec<char>,
        pos: usize,
    }

    pub fn parse(text: &str) -> Result<Value, String> {
        let mut p = Parser { chars: text.chars().collect(), pos: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.pos != p.chars.len() {
            return Err(format!("Extra data at char {}", p.pos));
        }
        Ok(v)
    }

    impl Parser {
        fn ws(&mut self) {
            while self.pos < self.chars.len() && matches!(self.chars[self.pos], ' ' | '\t' | '\n' | '\r') {
                self.pos += 1;
            }
        }

        fn peek(&self) -> Option<char> {
            self.chars.get(self.pos).copied()
        }

        fn expect(&mut self, c: char) -> Result<(), String> {
            if self.peek() == Some(c) {
                self.pos += 1;
                Ok(())
            } else {
                Err(format!("Expecting '{}' at char {}", c, self.pos))
            }
        }

        fn literal(&mut self, word: &str, v: Value) -> Result<Value, String> {
            for c in word.chars() {
                self.expect(c)?;
            }
            Ok(v)
        }

        fn value(&mut self) -> Result<Value, String> {
            match self.peek() {
                Some('{') => self.object(),
                Some('[') => self.array(),
                Some('"') => Ok(Value::Str(self.string()?)),
                Some('t') => self.literal("true", Value::Bool(true)),
                Some('f') => self.literal("false", Value::Bool(false)),
                Some('n') => self.literal("null", Value::Null),
                Some(c) if c == '-' || c.is_ascii_digit() => self.number(),
                _ => Err(format!("Expecting value at char {}", self.pos)),
            }
        }

        fn number(&mut self) -> Result<Value, String> {
            let start = self.pos;
            while self.pos < self.chars.len()
                && matches!(self.chars[self.pos], '-' | '+' | '.' | 'e' | 'E' | '0'..='9')
            {
                self.pos += 1;
            }
            let s: String = self.chars[start..self.pos].iter().collect();
            s.parse::<f64>().map(Value::Num).map_err(|_| format!("Bad number at char {}", start))
        }

        fn hex4(&mut self) -> Result<u32, String> {
            let mut n = 0u32;
            for _ in 0..4 {
                let c = self.peek().ok_or("Truncated \\u escape")?;
                n = n * 16 + c.to_digit(16).ok_or_else(|| format!("Bad \\u escape at char {}", self.pos))?;
                self.pos += 1;
            }
            Ok(n)
        }

        fn string(&mut self) -> Result<String, String> {
            self.expect('"')?;
            let mut out = String::new();
            loop {
                let c = self.peek().ok_or("Unterminated string")?;
                self.pos += 1;
                match c {
                    '"' => return Ok(out),
                    '\\' => {
                        let e = self.peek().ok_or("Unterminated string")?;
                        self.pos += 1;
                        match e {
                            '"' => out.push('"'),
                            '\\' => out.push('\\'),
                            '/' => out.push('/'),
                            'b' => out.push('\u{8}'),
                            'f' => out.push('\u{c}'),
                            'n' => out.push('\n'),
                            'r' => out.push('\r'),
                            't' => out.push('\t'),
                            'u' => {
                                let hi = self.hex4()?;
                                let cp = if (0xD800..0xDC00).contains(&hi)
                                    && self.peek() == Some('\\')
                                    && self.chars.get(self.pos + 1) == Some(&'u')
                                {
                                    self.pos += 2;
                                    let lo = self.hex4()?;
                                    0x10000 + ((hi - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF)
                                } else {
                                    hi
                                };
                                out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            }
                            _ => return Err(format!("Invalid escape at char {}", self.pos)),
                        }
                    }
                    c if (c as u32) < 0x20 => return Err(format!("Control character in string at char {}", self.pos)),
                    c => out.push(c),
                }
            }
        }

        fn array(&mut self) -> Result<Value, String> {
            self.expect('[')?;
            let mut items = Vec::new();
            self.ws();
            if self.peek() == Some(']') {
                self.pos += 1;
                return Ok(Value::Arr(items));
            }
            loop {
                self.ws();
                items.push(self.value()?);
                self.ws();
                match self.peek() {
                    Some(',') => self.pos += 1,
                    Some(']') => {
                        self.pos += 1;
                        return Ok(Value::Arr(items));
                    }
                    _ => return Err(format!("Expecting ',' or ']' at char {}", self.pos)),
                }
            }
        }

        fn object(&mut self) -> Result<Value, String> {
            self.expect('{')?;
            let mut items = Vec::new();
            self.ws();
            if self.peek() == Some('}') {
                self.pos += 1;
                return Ok(Value::Obj(items));
            }
            loop {
                self.ws();
                let key = self.string()?;
                self.ws();
                self.expect(':')?;
                self.ws();
                items.push((key, self.value()?));
                self.ws();
                match self.peek() {
                    Some(',') => self.pos += 1,
                    Some('}') => {
                        self.pos += 1;
                        return Ok(Value::Obj(items));
                    }
                    _ => return Err(format!("Expecting ',' or '}}' at char {}", self.pos)),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gemma: payload, reply parsing, truncation
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq)]
pub struct Payload {
    pub system_instruction: String,
    pub pass_1_blueprint: String,
    pub pass_1_tags: String,
    pub pass_2_blueprint: String,
    pub pass_2_tags: String,
}

pub fn build_payload(
    raw_wd14_tags: &str,
    pass1_txt: &str,
    pass2_txt: &str,
    mode: Mode,
    vocab: Option<&HashSet<String>>,
    vision: bool,
) -> Payload {
    let m = merge_tags(raw_wd14_tags, pass1_txt, pass2_txt, mode, vocab);
    let instruction = if vision {
        format!("{} {}", mode.llm_instruction(), VISION_TAIL)
    } else {
        mode.llm_instruction().to_string()
    };
    let pass2: Vec<String> = m.subject_tags.iter().chain(m.pass2_tags.iter()).cloned().collect();
    Payload {
        system_instruction: instruction,
        pass_1_blueprint: flatten_zones(pass1_txt),
        pass_1_tags: m.pass1_tags.join(", "),
        pass_2_blueprint: flatten_zones(pass2_txt),
        pass_2_tags: pass2.join(", "),
    }
}

/// Gemma has no system role: the instruction goes at the top of the user turn.
pub fn payload_to_text(p: &Payload) -> String {
    format!(
        "{}\n\nINPUT:\n{{\n  \"pass_1_blueprint\": \"{}\",\n  \"pass_1_tags\": \"{}\",\n  \"pass_2_blueprint\": \"{}\",\n  \"pass_2_tags\": \"{}\"\n}}",
        p.system_instruction,
        json::escape(&p.pass_1_blueprint),
        json::escape(&p.pass_1_tags),
        json::escape(&p.pass_2_blueprint),
        json::escape(&p.pass_2_tags),
    )
}

fn strip_fences(raw: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    for line in raw.trim().split('\n') {
        let mut l = line;
        if l.starts_with("```") {
            l = &l[3..];
            if l.len() >= 4 && l.is_char_boundary(4) && l[..4].eq_ignore_ascii_case("json") {
                l = &l[4..];
            }
            l = l.trim_start();
        }
        if l.ends_with("```") {
            l = l[..l.len() - 3].trim_end();
        }
        lines.push(l.to_string());
    }
    lines.join("\n").trim().to_string()
}

fn py_str(v: &json::Value) -> Option<String> {
    match v {
        json::Value::Str(s) => Some(s.clone()),
        json::Value::Num(n) => Some(if n.fract() == 0.0 && n.abs() < 1e15 {
            format!("{}", *n as i64)
        } else {
            format!("{}", n)
        }),
        json::Value::Bool(b) => Some(if *b { "True".into() } else { "False".into() }),
        json::Value::Null => Some("None".into()),
        _ => None,
    }
}

fn as_str(v: Option<&json::Value>) -> String {
    match v {
        Some(json::Value::Str(s)) => s.trim().to_string(),
        Some(json::Value::Arr(items)) => items
            .iter()
            .filter_map(py_str)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(", "),
        _ => String::new(),
    }
}

/// `Some((scene, character))` (either may be empty) or `None` if unparseable.
pub fn try_parse_gemma_json(raw: &str) -> Option<(String, String)> {
    let text = strip_fences(raw);
    let candidate = match (text.find('{'), text.rfind('}')) {
        (Some(s), Some(e)) if e > s => &text[s..=e],
        _ => text.as_str(),
    };
    match json::parse(candidate) {
        Ok(v @ json::Value::Obj(_)) => {
            Some((as_str(v.get("scene_prompt")), as_str(v.get("character_prompt"))))
        }
        _ => None,
    }
}

/// Never fails; on bad output returns a best-effort fallback.
pub fn parse_gemma_json_response(raw: &str, cfg: &Config) -> Prompts {
    if let Some((scene, character)) = try_parse_gemma_json(raw) {
        return Prompts {
            scene_prompt: if scene.is_empty() { cfg.quality_prefix.clone() } else { scene },
            character_prompt: character,
        };
    }
    let text = strip_fences(raw);
    let stripped: String = text.chars().filter(|c| !matches!(c, '{' | '}' | '"')).collect();
    Prompts {
        scene_prompt: cfg.quality_prefix.clone(),
        character_prompt: clean_csv(&[&stripped.replace('\n', ", ")]),
    }
}

pub struct GemmaRequest<'a> {
    pub text: &'a str,
    pub temperature: f32,
    pub max_tokens: u32,
    /// True on the vision pass: your closure attaches the image.
    pub with_image: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GemmaReply {
    pub text: String,
    /// Ollama's `done_reason`; "length" means the reply hit the token cap.
    pub finish_reason: Option<String>,
}

/// True if the reply was cut off. With no finish reason, a reply that does not end
/// in `}` (ignoring a closing code fence) counts as cut off.
pub fn gemma_truncated(reply: &GemmaReply) -> bool {
    if let Some(r) = &reply.finish_reason {
        return matches!(
            r.trim().to_lowercase().as_str(),
            "length" | "max_tokens" | "max_length" | "max_new_tokens"
        );
    }
    let t = reply.text.trim();
    let t = t.strip_suffix("```").map_or(t, |x| x.trim_end());
    !t.ends_with('}')
}

// ---------------------------------------------------------------------------
// Ollama helpers (you supply the HTTP transport)
// ---------------------------------------------------------------------------
/// Body for `POST {host}/api/chat`. `image_b64` is the base64 image on the vision pass.
pub fn ollama_chat_request_body(
    model: &str,
    text: &str,
    temperature: f32,
    num_ctx: u32,
    num_predict: u32,
    image_b64: Option<&str>,
    json_mode: bool,
) -> String {
    let images = image_b64
        .map(|b| format!(",\"images\":[\"{}\"]", json::escape(b)))
        .unwrap_or_default();
    let format = if json_mode { ",\"format\":\"json\"" } else { "" };
    format!(
        "{{\"model\":\"{}\",\"messages\":[{{\"role\":\"user\",\"content\":\"{}\"{}}}],\"stream\":false,\"options\":{{\"temperature\":{},\"num_ctx\":{},\"num_predict\":{}}}{}}}",
        json::escape(model),
        json::escape(text),
        images,
        temperature,
        num_ctx,
        num_predict,
        format
    )
}

/// Parse the body returned by `/api/chat` (non-streaming).
pub fn parse_ollama_chat_response(body: &str) -> Result<GemmaReply, String> {
    let v = json::parse(body)?;
    let content = match v.get("message").and_then(|m| m.get("content")) {
        Some(json::Value::Str(s)) => s.clone(),
        _ => String::new(),
    };
    let reason = match v.get("done_reason") {
        Some(json::Value::Str(s)) => Some(s.clone()),
        _ => None,
    };
    Ok(GemmaReply { text: content, finish_reason: reason })
}

/// Is `model` pulled? `tags_body` is the response of `GET {host}/api/tags`.
pub fn ollama_model_pulled(tags_body: &str, model: &str) -> Result<bool, String> {
    let v = json::parse(tags_body)?;
    let mut wanted = vec![model.to_string()];
    if !model.contains(':') {
        wanted.push(format!("{}:latest", model));
    }
    if let Some(json::Value::Arr(models)) = v.get("models") {
        for m in models {
            for key in ["name", "model"] {
                if let Some(json::Value::Str(n)) = m.get(key) {
                    if wanted.contains(n) {
                        return Ok(true);
                    }
                }
            }
        }
    }
    Ok(false)
}

pub fn ollama_not_pulled_message(model: &str) -> String {
    format!("Model {} is not pulled in Ollama; run `ollama pull {}`", model, model)
}

// ---------------------------------------------------------------------------
// VLM retry wrapper
// ---------------------------------------------------------------------------
fn valid_vlm(text: &str) -> bool {
    let t = text.trim();
    if t.chars().count() < 20 {
        return false;
    }
    let head: String = t.chars().take(80).collect::<String>().to_lowercase();
    !["sorry", "cannot", "can't", "unable to"]
        .iter()
        .any(|p| contains_whole_phrase(&head, p))
}

#[inline]
fn valid_moondream(text: &str) -> bool {
    valid_vlm(text)
}

/// One VLM pass: retry once if it errors, is empty or looks like a refusal.
/// Gives up with "" (that pass then contributes only its WD14 tags) plus a warning.
fn run_vlm(
    vlm: &VlmFn<'_>,
    prompt: &str,
    label: &str,
    cfg: &Config,
    warnings: &mut Vec<String>,
) -> String {
    let mut last_error: Option<String> = None;
    for attempt in 0..=cfg.vlm_retries {
        let temp = cfg.vlm_temperature + if attempt > 0 { cfg.vlm_retry_temp_bump } else { 0.0 };
        match vlm(prompt, temp) {
            Ok(out) if valid_vlm(&out) => return out,
            Ok(_) => {}
            Err(e) => last_error = Some(e),
        }
    }
    let reason = last_error.unwrap_or_else(|| "empty or refusal-like output".to_string());
    warnings.push(format!("VLM {} unavailable ({}); used WD14 tags only for it", label, reason));
    String::new()
}

#[inline]
fn run_moondream(
    vlm: &VlmFn<'_>,
    prompt: &str,
    label: &str,
    cfg: &Config,
    warnings: &mut Vec<String>,
) -> String {
    run_vlm(vlm, prompt, label, cfg, warnings)
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------
/// Runs one VLM pass on the image: `(prompt, temperature) -> text`.
pub type VlmFn<'a> = dyn Fn(&str, f32) -> Result<String, String> + 'a;
pub type MoondreamFn<'a> = VlmFn<'a>;
/// Runs one Gemma request (attach the image when `with_image` is true).
pub type GemmaFn<'a> = dyn Fn(&GemmaRequest) -> Result<GemmaReply, String> + 'a;

#[derive(Clone, Copy, Debug)]
pub struct PipelineOptions {
    pub mode: Mode,
    pub use_llm_synthesis: bool,
    /// Only applies when `use_llm_synthesis` is true.
    pub gemma_vision_pass: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PipelineResult {
    pub scene_prompt: String,
    pub character_prompt: String,
    /// What fell back and why. Show this in the UI: a silent fallback looks like
    /// a normal result otherwise. Empty when nothing went wrong.
    pub warnings: Vec<String>,
}

/// Run the dual-pass pipeline.
///
/// * `vlm(prompt, temperature)` runs one VLM pass on the image.
/// * `gemma` is `None` when no Gemma backend is reachable (e.g. your Ollama health
///   check failed): the mechanical merge is used and a warning recorded.
/// * Never fails: every error becomes a warning plus a fallback.
pub fn run_dual_pass(
    raw_wd14_tags: &str,
    opts: &PipelineOptions,
    cfg: &Config,
    vocab: Option<&HashSet<String>>,
    vlm: &VlmFn<'_>,
    gemma: Option<&GemmaFn<'_>>,
) -> PipelineResult {
    let mode = opts.mode;
    let mut warnings: Vec<String> = Vec::new();
    let pass1 = run_vlm(vlm, mode.pass1_prompt(), "pass 1", cfg, &mut warnings);
    let pass2 = run_vlm(vlm, mode.pass2_prompt(), "pass 2", cfg, &mut warnings);

    let mechanical = |reason: Option<String>, mut warnings: Vec<String>| {
        if let Some(r) = reason {
            warnings.push(r);
        }
        let p = process_direct(raw_wd14_tags, &pass1, &pass2, mode, vocab, cfg);
        PipelineResult { scene_prompt: p.scene_prompt, character_prompt: p.character_prompt, warnings }
    };

    if !opts.use_llm_synthesis {
        let note = opts
            .gemma_vision_pass
            .then(|| "gemma_vision_pass ignored: use_llm_synthesis is off".to_string());
        return mechanical(note, warnings);
    }
    let gemma = match gemma {
        Some(g) => g,
        None => {
            return mechanical(
                Some("No Gemma backend available; used the mechanical merge".to_string()),
                warnings,
            )
        }
    };

    let payload = build_payload(raw_wd14_tags, &pass1, &pass2, mode, vocab, opts.gemma_vision_pass);
    let text = payload_to_text(&payload);
    let call = |max_tokens: u32| {
        gemma(&GemmaRequest {
            text: &text,
            temperature: cfg.gemma_temperature,
            max_tokens,
            with_image: opts.gemma_vision_pass,
        })
    };
    let mut reply = match call(cfg.gemma_max_tokens) {
        Ok(r) => r,
        Err(e) => {
            return mechanical(Some(format!("Gemma call failed ({}); used the mechanical merge", e)), warnings)
        }
    };
    if gemma_truncated(&reply) {
        // One retry with a higher cap. A still-truncated reply fails to parse below
        // and ends up in the "output unusable" fallback.
        reply = match call(cfg.gemma_max_tokens_retry) {
            Ok(r) => r,
            Err(e) => {
                return mechanical(Some(format!("Gemma call failed ({}); used the mechanical merge", e)), warnings)
            }
        };
    }

    match try_parse_gemma_json(&reply.text) {
        Some((scene, character)) if !character.is_empty() => PipelineResult {
            scene_prompt: style_tags(
                &if scene.is_empty() { cfg.quality_prefix.clone() } else { scene },
                cfg.escape_parentheses,
            ),
            character_prompt: style_tags(&character, cfg.escape_parentheses),
            warnings,
        },
        _ => mechanical(Some("Gemma output unusable; used the mechanical merge".to_string()), warnings),
    }
}

// ---------------------------------------------------------------------------
// Phase 2: Pure Crop Tag Policy & Zone Math
// ---------------------------------------------------------------------------

pub fn is_crop_artifact_tag(tag: &str) -> bool {
    let norm = norm_tag(tag);
    if CROP_ARTIFACT_TAGS.iter().any(|&c| c == norm) {
        return true;
    }
    if norm.ends_with(" focus") {
        return true;
    }
    if norm.ends_with(" out of frame") {
        return true;
    }
    if norm.starts_with("cropped ") || norm.ends_with(" cropped") {
        return true;
    }
    false
}

pub fn is_rating_tag(tag: &str) -> bool {
    let norm = norm_tag(tag);
    RATING_TAGS.iter().any(|&r| r == norm)
}

pub fn is_body_size_or_count_tag(tag: &str) -> bool {
    let norm = norm_tag(tag);
    BODY_SIZE_AND_COUNT_TAGS.iter().any(|&b| b == norm)
}

pub fn is_concrete_scene_object(tag: &str) -> bool {
    let norm = norm_tag(tag);
    CONCRETE_SCENE_OBJECTS.iter().any(|&obj| norm == obj || norm.contains(obj))
}

pub fn is_global_framing_tag(tag: &str) -> bool {
    let norm = norm_tag(tag);
    if is_crop_artifact_tag(&norm) {
        return true;
    }
    if SCENE_KEYWORDS.iter().any(|&k| k == norm) {
        return true;
    }
    norm.ends_with(" shot") || norm.ends_with(" view") || norm.ends_with(" angle") || norm.ends_with(" background")
}

pub fn filter_crop_tags(
    crop_tags: &[(String, f32)],
    crop_type: CropType,
    character_names: &[String],
    min_attribute_threshold: f32,
    scene_object_threshold: f32,
) -> Vec<(String, f32)> {
    let char_set: HashSet<String> = character_names.iter().map(|c| norm_tag(c)).collect();
    let mut accepted = Vec::new();

    for (tag, score) in crop_tags {
        let norm = norm_tag(tag);
        if norm.is_empty() {
            continue;
        }

        // 1. Reject crop artifacts (boundary & zoom artifacts)
        if is_crop_artifact_tag(&norm) {
            continue;
        }

        // 2. Reject ratings
        if is_rating_tag(&norm) {
            continue;
        }

        // 3. Reject character names
        if char_set.contains(&norm) {
            continue;
        }

        // 4. Reject body size & count tags
        if is_body_size_or_count_tag(&norm) {
            continue;
        }

        match crop_type {
            CropType::CharacterZone => {
                // Reject global framing/scene
                if is_global_framing_tag(&norm) {
                    continue;
                }
                // Micro-attributes require confidence >= 0.55
                if *score >= min_attribute_threshold {
                    accepted.push((tag.clone(), *score));
                }
            }
            CropType::Quadrant => {
                // Reject generic backgrounds
                if norm == "simple background"
                    || norm == "white background"
                    || norm == "grey background"
                    || norm == "monochrome"
                    || norm == "blurry background"
                {
                    continue;
                }
                if is_concrete_scene_object(&norm) {
                    if *score >= scene_object_threshold {
                        accepted.push((tag.clone(), *score));
                    }
                } else if !is_global_framing_tag(&norm) && *score >= min_attribute_threshold {
                    accepted.push((tag.clone(), *score));
                }
            }
        }
    }

    accepted
}

pub fn merge_crop_tags(
    base_tags: &[(String, f32)],
    crop_tags: &[(String, f32)],
    crop_type: CropType,
    character_names: &[String],
    min_attribute_threshold: f32,
    scene_object_threshold: f32,
) -> Vec<(String, f32)> {
    let filtered = filter_crop_tags(
        crop_tags,
        crop_type,
        character_names,
        min_attribute_threshold,
        scene_object_threshold,
    );

    let mut out: Vec<(String, f32)> = base_tags.to_vec();

    for (c_tag, c_score) in filtered {
        let norm_c = norm_tag(&c_tag);
        let mut found = false;
        for (b_tag, b_score) in out.iter_mut() {
            if norm_tag(b_tag) == norm_c {
                *b_score = b_score.max(c_score);
                found = true;
                break;
            }
        }
        if !found {
            out.push((c_tag, c_score));
        }
    }
    out
}

pub fn compute_zone_boxes(
    img_w: f32,
    img_h: f32,
    box_x0: f32,
    box_y0: f32,
    box_x1: f32,
    box_y1: f32,
) -> Vec<ZoneBox> {
    if img_w <= 0.0 || img_h <= 0.0 {
        return Vec::new();
    }

    let x0 = box_x0.max(0.0).min(img_w);
    let y0 = box_y0.max(0.0).min(img_h);
    let x1 = box_x1.max(x0).min(img_w);
    let y1 = box_y1.max(y0).min(img_h);

    let w = x1 - x0;
    let h = y1 - y0;
    if w <= 1.0 || h <= 1.0 {
        return Vec::new();
    }

    let bottom_edge = y1 >= (0.98 * img_h);
    let height_ratio = h / img_h;

    let mx = w * 0.06;
    let my = h * 0.04;

    let make_box = |name: &'static str, zy0: f32, zy1: f32| -> ZoneBox {
        let cx0 = (x0 - mx).max(0.0).min(img_w - 1.0);
        let cy0 = (zy0 - my).max(0.0).min(img_h - 1.0);
        let cx1 = (x1 + mx).min(img_w).max(cx0 + 1.0);
        let cy1 = (zy1 + my).min(img_h).max(cy0 + 1.0);
        ZoneBox {
            name,
            x0: (cx0 * 100.0).round() / 100.0,
            y0: (cy0 * 100.0).round() / 100.0,
            x1: (cx1 * 100.0).round() / 100.0,
            y1: (cy1 * 100.0).round() / 100.0,
        }
    };

    let mut zones = Vec::new();
    // 1. Upper (0 - 55%): Head, Hair, Face - Always valid
    zones.push(make_box("Upper", y0, y0 + h * 0.55));

    // 2. Middle (30% - 70%): Torso, Arms, Held Items
    if height_ratio >= 0.25 {
        zones.push(make_box("Middle", y0 + h * 0.30, y0 + h * 0.70));
    }

    // 3. Lower (55% - 100%): Hips, Legs, Footwear
    // Skip lower zone if figure touches the bottom frame and height ratio < 0.72 (half-body/cowboy shot cut off)
    let is_cut_off_bottom = bottom_edge && (height_ratio < 0.72);
    if !is_cut_off_bottom && (height_ratio >= 0.40) {
        zones.push(make_box("Lower", y0 + h * 0.55, y1));
    }

    zones
}
