fn main() {
    // element debug info: the window smoke test finds buttons by label and
    // checks they stay inside the window (`window_smoke::assert_reachable`).
    // Required whenever tests run, including `cargo test --release`.
    let config = slint_build::CompilerConfiguration::new()
        .with_style("material-dark".into())
        .with_debug_info(true);
    slint_build::compile_with_config("ui/app.slint", config).unwrap();
}
