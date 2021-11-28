extern crate env_logger;
extern crate renegade;
extern crate tracing;

use std::{
    collections::{HashMap, HashSet},
    io::BufRead,
    path::PathBuf,
};

use log::LevelFilter;
use renegade::{metric::learn_metrics, LearnerConfig};
use tracing::*;
use xz::read::XzDecoder;

#[derive(Debug)]
struct Embedding {
    word: String,
    vector: Vec<f64>,
}

#[test]
fn word2vec_test() {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(LevelFilter::Info)
        .try_init();

    let config = LearnerConfig {
        sample_count: 10000,
        train_test_prop: 0.5,
        iterations: 10,
        learning_rate: 0.01,
    };

    let counties: HashMap<&'static str, i32> = get_counties();

    let mut test_set = HashSet::new();
    test_set.insert("Meath");

    let mut embedding_map: HashMap<String, Vec<f64>> = HashMap::new();
    for embedding in read_embeddings() {
        embedding_map.insert(embedding.word, embedding.vector);
    }

    println!("Load complete");

    let county_embeddings: Vec<(Vec<f64>, i32)> = counties
        .iter()
        .map(|(county, population)| (embedding_map[*county].clone(), *population))
        .collect();

    learn_metrics::<Vec<f64>, i32>(&county_embeddings, input_metrics, output_metric, &config);
}

//   let input_metrics = |a: &Vec<f64>, b: &Vec<f64>| a.iter().zip(b.iter()).map(|(a, b)| (a - b).abs()).collect();
//   let output_metric = |a: i32, b: i32| (a - b).abs();

fn input_metrics(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(a, b)| (a - b).abs()).collect()
}

fn output_metric(a: &i32, b: &i32) -> f64 {
    (a - b).abs() as f64
}

fn get_counties() -> HashMap<&'static str, i32> {
    HashMap::from([
        ("Dublin", 1345402),
        ("Antrim", 618108),
        ("Cork", 542868),
        ("Down", 531665),
        ("Fingal", 296020),
        ("South Dublin", 278767),
        ("Galway", 258058),
        ("Londonderry", 247132),
        ("Kildare", 222504),
        ("Dún Laoghaire–Rathdown", 218020),
        ("Meath", 195044),
        ("Limerick", 194999),
        ("Tyrone", 179000),
        ("Armagh", 174792),
        ("Tipperary", 159553),
        ("Donegal", 159192),
        ("Wexford", 149722),
        ("Kerry", 147707),
        ("Wicklow", 142425),
        ("Mayo", 130507),
        ("Louth", 128884),
        ("Clare", 118017),
        ("Waterford", 116176),
        ("Kilkenny", 99230),
        ("Westmeath", 88771),
        ("Laois", 84697),
        ("Offaly", 77961),
        ("Cavan", 76176),
        ("Sligo", 65535),
        ("Roscommon", 64544),
        ("Monaghan", 61386),
        ("Fermanagh", 61170),
        ("Carlow", 56932),
        ("Longford", 40873),
        ("Leitrim", 32044),
    ])
}

fn read_embeddings() -> impl Iterator<Item = Embedding> {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("tests/resources/model.txt.xz");
    let decoder = XzDecoder::new(std::fs::File::open(d).unwrap());

    std::io::BufReader::new(decoder)
        .lines()
        .skip(1)
        .filter_map(|line| {
            let line = line.unwrap();
            let split: Vec<&str> = line.split_whitespace().collect();
            if split.len() != 301 {
                None
            } else {
                let word = split[0];
                let array: Vec<f64> = split
                    .iter()
                    .skip(1)
                    .map(|x| x.parse::<f64>().unwrap())
                    .collect();
                Some(Embedding {
                    word: word.to_string(),
                    vector: array,
                })
            }
        })
}
