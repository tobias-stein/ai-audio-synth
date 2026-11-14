// The library directory is embedded in the binary; rebuild when it changes.
fn main() {
    println!("cargo:rerun-if-changed=library");
    println!("cargo:rerun-if-changed=docs/AGENT_GUIDE.md");
}
