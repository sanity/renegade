extern crate pav_regression;
extern crate bit_vec;

mod renegade;

#[cfg(test)]
mod tests {

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn it_works() {
        init();
        assert_eq!(2 + 2, 4);
    }
}
