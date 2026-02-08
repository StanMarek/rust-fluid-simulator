// Non-WASM entry point: just print a message.
// The actual WASM entry is in lib.rs.
fn main() {
    eprintln!("This binary is intended for WASM. Use `apps/desktop` for native.");
    eprintln!("Build with: cd apps/web && ./build.sh");
}
