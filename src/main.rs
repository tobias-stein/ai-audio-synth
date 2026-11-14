mod analysis;
mod audio;
mod cli;
mod doc;
mod dsp;
mod error;
mod expr;
mod graph;
mod library;
mod nodes;
mod score;
mod theory;
mod render;
mod util;
mod wav;

fn main() {
    std::process::exit(cli::run());
}
