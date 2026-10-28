use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // The generated header is written to OUT_DIR. The committed copy at the repository
    // root is refreshed from there, so the build never edits the source tree.
    cbindgen::generate(&crate_dir)
        .expect("Failed to generate C bindings")
        .write_to_file(out_dir.join("tpt_raglite.h"));
}
