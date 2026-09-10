use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(&crate_dir).join("..");

    cbindgen::generate(&crate_dir)
        .expect("Failed to generate C bindings")
        .write_to_file(out_dir.join("tpt_raglite.h"));
}
