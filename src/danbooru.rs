//! Danbooru tag list (the WD tagger's `selected_tags.csv`, ~10k common tags with
//! usage counts). Used by the Danbooru translator to offer real tag names to the AI
//! and to check the tags it writes.

use anyhow::{anyhow, Context, Result};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct TagInfo {
    /// Prompt form: spaces instead of underscores ("holding umbrella").
    pub name: String,
    pub count: u64,
    pub character: bool,
}

pub struct TagList {
    tags: Vec<TagInfo>,
    by_name: HashMap<String, usize>,
}

static LIST: OnceLock<TagList> = OnceLock::new();

fn csv_path() -> std::path::PathBuf {
    crate::tagger::tagger_dir().join("selected_tags.csv")
}

/// Lowercase, underscores to spaces, escaped parentheses unescaped.
pub fn key(tag: &str) -> String {
    tag.trim()
        .to_lowercase()
        .replace("\\(", "(")
        .replace("\\)", ")")
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse(text: &str) -> TagList {
    let mut tags = Vec::new();
    // tag_id,name,category,count (names may contain commas)
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 4 {
            continue;
        }
        let category = f[f.len() - 2];
        if category == "9" {
            continue; // rating tags
        }
        tags.push(TagInfo {
            name: key(&f[1..f.len() - 2].join(",")),
            count: f[f.len() - 1].trim().parse().unwrap_or(0),
            character: category == "4",
        });
    }
    let by_name = tags.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
    TagList { tags, by_name }
}

/// The tag list, downloading the small CSV (~300 KB) the first time if the tagger
/// isn't installed.
pub fn list() -> Result<&'static TagList> {
    if let Some(l) = LIST.get() {
        return Ok(l);
    }
    let path = csv_path();
    if !path.is_file() {
        std::fs::create_dir_all(path.parent().expect("has a parent"))?;
        let body = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()?
            .get(format!("{}/selected_tags.csv", crate::tagger::REPO))
            .header("User-Agent", "prompt-condenser")
            .send()
            .context("Couldn't download the Danbooru tag list")?;
        if !body.status().is_success() {
            return Err(anyhow!("Downloading the Danbooru tag list failed: HTTP {}", body.status()));
        }
        std::fs::write(&path, body.bytes()?)?;
    }
    let text = std::fs::read_to_string(&path)?;
    Ok(LIST.get_or_init(|| parse(&text)))
}

const STOPWORDS: &[&str] = &[
    "the", "and", "with", "her", "his", "their", "she", "him", "they", "that", "this", "from", "into", "onto",
    "over", "under", "while", "who", "which", "very", "some", "for", "are", "was", "is", "its", "has", "have",
    "being", "wearing", "looking", "a", "an", "of", "in", "on", "at", "to", "by", "as", "or", "but", "out",
];

impl TagList {
    pub fn get(&self, tag: &str) -> Option<&TagInfo> {
        self.by_name.get(&key(tag)).map(|&i| &self.tags[i])
    }

    /// Real tag for a near miss: plural/singular or a dash/space difference.
    pub fn fix(&self, tag: &str) -> Option<&TagInfo> {
        let k = key(tag);
        let tries = [
            format!("{k}s"),
            k.strip_suffix('s').unwrap_or(&k).to_string(),
            k.replace('-', " "),
            k.replace(' ', "-"),
        ];
        tries.iter().find_map(|t| self.get(t))
    }

    /// Real tags close to an unknown one, ranked by its distinctive words: words shared
    /// with hundreds of tags ("eyes", "hair") don't count, so "downcast eyes" doesn't
    /// suggest "blue eyes".
    pub fn similar(&self, tag: &str, n: usize) -> Vec<&TagInfo> {
        let words = |t: &str| -> Vec<String> {
            t.split(|c: char| !c.is_alphanumeric())
                .filter(|w| w.len() >= 3 && !STOPWORDS.contains(w))
                .map(|w| w.strip_suffix('s').filter(|s| s.len() >= 3).unwrap_or(w).to_string())
                .collect()
        };
        let want = words(&key(tag));
        let freq = |w: &str| self.tags.iter().filter(|t| !t.character && words(&t.name).iter().any(|x| x == w)).count();
        let weights: Vec<(String, f32)> = want
            .iter()
            .filter_map(|w| {
                let f = freq(w);
                (f > 0 && f <= 40).then(|| (w.clone(), 1.0 / f as f32))
            })
            .collect();
        if weights.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(f32, &TagInfo)> = self
            .tags
            .iter()
            .filter(|t| !t.character)
            .filter_map(|t| {
                let tw = words(&t.name);
                let score: f32 = weights.iter().filter(|(w, _)| tw.contains(w)).map(|(_, x)| x).sum();
                (score > 0.0).then_some((score, t))
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.count.cmp(&a.1.count)));
        scored.into_iter().take(n).map(|(_, t)| t).collect()
    }

    /// Common tags that contain the description's words, most used first.
    pub fn candidates(&self, text: &str, per_word: usize, total: usize) -> Vec<&TagInfo> {
        let words: Vec<String> = text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3 && !STOPWORDS.contains(w))
            .map(str::to_string)
            .collect();
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for w in &words {
            let stem = w.strip_suffix("es").filter(|s| s.len() >= 3).or_else(|| w.strip_suffix('s')).unwrap_or(w);
            let mut hits: Vec<&TagInfo> = self
                .tags
                .iter()
                .filter(|t| !t.character)
                .filter(|t| t.name.split(|c: char| !c.is_alphanumeric()).any(|tw| tw == w || tw == stem || tw.strip_suffix('s') == Some(stem)))
                .collect();
            hits.sort_by(|a, b| b.count.cmp(&a.count));
            for t in hits.into_iter().take(per_word) {
                if seen.insert(t.name.clone()) {
                    out.push(t);
                }
            }
            if out.len() >= total {
                break;
            }
        }
        out.truncate(total);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_and_fixes_tags() {
        let l = parse(
            "tag_id,name,category,count\n1,holding_umbrella,0,50000\n2,umbrella,0,90000\n3,rain,0,80000\n\
             4,hatsune_miku,4,100000\n5,general,9,1\n6,ribbon,0,70000\n",
        );
        assert_eq!(l.get("Holding_Umbrella").unwrap().count, 50000);
        assert!(l.get("general").is_none());
        assert_eq!(l.fix("ribbons").unwrap().name, "ribbon");
        let c: Vec<&str> = l.candidates("a girl sheltering under umbrellas in the rain", 5, 20).iter().map(|t| t.name.as_str()).collect();
        assert_eq!(c, vec!["umbrella", "holding umbrella", "rain"]);
        let s: Vec<&str> = l.similar("holding parasol umbrella", 2).iter().map(|t| t.name.as_str()).collect();
        assert_eq!(s, vec!["holding umbrella", "umbrella"]);
    }
}
