extern crate pav_regression;

pub mod renegade;
#[cfg(test)]
mod tests {
    use super::*;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn it_works() {
        init();
        assert_eq!(2 + 2, 4);
    }
}
