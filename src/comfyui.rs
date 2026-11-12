//! Installs the ComfyUI add-on (`tools/comfyui`, embedded at build time) into
//! `<ComfyUI>/custom_nodes/ComfyUI-Prompt-Condenser`, with `binary_path.txt` pointing at this binary.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// The add-on folder's name under `custom_nodes`.
pub const ADDON_DIR: &str = "ComfyUI-Prompt-Condenser";

/// Paths inside the add-on folder (`js/` is the add-on's `WEB_DIRECTORY`).
const FILES: [(&str, &str); 4] = [
    ("__init__.py", include_str!("../tools/comfyui/__init__.py")),
    ("nodes.py", include_str!("../tools/comfyui/nodes.py")),
    ("README.md", include_str!("../tools/comfyui/README.md")),
    ("js/queue.js", include_str!("../tools/comfyui/js/queue.js")),
];

/// What an install did.
#[derive(Debug, PartialEq)]
pub enum Installed {
    /// The files were written to this folder.
    Written(PathBuf),
    /// The add-on folder is a symlink (a development install); nothing was written through it.
    AlreadyLinked { link: PathBuf, target: PathBuf },
}

impl Installed {
    /// The status line for the window and the CLI.
    pub fn message(&self) -> String {
        match self {
            Installed::Written(p) => format!("Installed to {}, restart ComfyUI", p.display()),
            Installed::AlreadyLinked { link, target } => {
                format!("{} is already linked to {}; left it alone", link.display(), target.display())
            }
        }
    }
}

/// Writes the add-on (its files plus `binary_path.txt` = this executable) into
/// `<comfyui>/custom_nodes/ComfyUI-Prompt-Condenser`. Refuses when `custom_nodes` is missing (not a
/// ComfyUI folder); never writes through a symlinked add-on folder.
pub fn install_comfyui_addon(comfyui: &Path) -> Result<Installed> {
    let exe = std::env::current_exe().context("can't tell where this program is")?;
    install_with_binary(comfyui, &exe)
}

fn install_with_binary(comfyui: &Path, binary: &Path) -> Result<Installed> {
    let nodes = comfyui.join("custom_nodes");
    if !nodes.is_dir() {
        bail!("{} has no custom_nodes folder; is it a ComfyUI folder?", comfyui.display());
    }
    let dir = nodes.join(ADDON_DIR);
    match std::fs::symlink_metadata(&dir) {
        Ok(meta) if meta.file_type().is_symlink() => {
            let target = std::fs::read_link(&dir).unwrap_or_default();
            return Ok(Installed::AlreadyLinked { link: dir, target });
        }
        Ok(meta) if !meta.is_dir() => bail!("{} exists and is not a folder", dir.display()),
        Ok(_) => {}
        Err(_) => std::fs::create_dir(&dir).with_context(|| format!("can't create {}", dir.display()))?,
    }
    for (name, text) in FILES {
        if let Some(parent) = dir.join(name).parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("can't create {}", parent.display()))?;
        }
        std::fs::write(dir.join(name), text).with_context(|| format!("can't write {}", dir.join(name).display()))?;
    }
    std::fs::write(dir.join("binary_path.txt"), binary.to_string_lossy().as_bytes())
        .with_context(|| format!("can't write {}", dir.join("binary_path.txt").display()))?;
    Ok(Installed::Written(dir))
}

/// `prompt-condenser --install-comfyui [path]`: the path, else the ComfyUI folder found the way the
/// Options' FOLDERS row finds it (`$COMFYUI_PATH`, else `~/ComfyUI`).
pub fn run_install_cli(args: &[String]) -> Result<()> {
    let given = args.first().map(String::as_str).unwrap_or("");
    let Some(folder) = crate::lora_meta::comfyui_folder(given) else {
        bail!("no ComfyUI folder found: pass one (--install-comfyui <path>) or set $COMFYUI_PATH");
    };
    println!("{}", install_comfyui_addon(&folder)?.message());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pc-comfyui-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn installs_the_files_and_the_binary_path() {
        let root = temp("install");
        std::fs::create_dir(root.join("custom_nodes")).unwrap();
        let r = install_with_binary(&root, Path::new("/opt/pc/prompt-condenser")).unwrap();
        let dir = root.join("custom_nodes").join(ADDON_DIR);
        assert_eq!(r, Installed::Written(dir.clone()));
        let mut names: Vec<String> =
            std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        assert_eq!(names, ["README.md", "__init__.py", "binary_path.txt", "js", "nodes.py"]);
        assert_eq!(std::fs::read_to_string(dir.join("js/queue.js")).unwrap(), FILES[3].1);
        assert_eq!(std::fs::read_to_string(dir.join("binary_path.txt")).unwrap(), "/opt/pc/prompt-condenser");
        assert_eq!(std::fs::read_to_string(dir.join("nodes.py")).unwrap(), FILES[1].1);
        // a second install updates in place
        assert!(matches!(install_with_binary(&root, Path::new("/usr/bin/prompt-condenser")).unwrap(), Installed::Written(_)));
        assert_eq!(std::fs::read_to_string(dir.join("binary_path.txt")).unwrap(), "/usr/bin/prompt-condenser");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_symlinked_addon_is_left_alone() {
        let root = temp("link");
        let repo = root.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir(root.join("custom_nodes")).unwrap();
        let link = root.join("custom_nodes").join(ADDON_DIR);
        std::os::unix::fs::symlink(&repo, &link).unwrap();
        let r = install_with_binary(&root, Path::new("/x")).unwrap();
        assert_eq!(r, Installed::AlreadyLinked { link: link.clone(), target: repo.clone() });
        assert!(r.message().contains("already linked to"));
        assert_eq!(std::fs::read_dir(&repo).unwrap().count(), 0, "nothing written through the link");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_folder_without_custom_nodes_is_refused() {
        let root = temp("refuse");
        let err = install_with_binary(&root, Path::new("/x")).unwrap_err();
        assert!(err.to_string().contains("custom_nodes"));
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
