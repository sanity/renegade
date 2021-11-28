extern crate renegade;
extern crate env_logger;
extern crate tracing;

use tracing::*;
use log::LevelFilter;

fn init() {
    let _ = env_logger::builder().is_test(true).filter_level(LevelFilter::Debug).try_init();
}

#[test]
fn waypoint_index_test() {
    init();

    let _span = span!(
        Level::INFO,
        "test_span",
    )
    .entered();
    event!(Level::INFO, test_val = 15,  "info event");
}