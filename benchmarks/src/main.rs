use std::collections::HashMap;
use std::fmt::Write as _;

use arroy::distances::Cosine;
use benchmarks::scenarios::ScenarioSearch;
use benchmarks::{arroy_bench, scenarios, MatLEView, RNG_SEED};
use byte_unit::Byte;
use clap::Parser;
use enum_iterator::Sequence;
use itertools::{iproduct, Itertools};
use ordered_float::OrderedFloat;
use rand::rngs::StdRng;
use rand::seq::SliceRandom as _;
use rand::SeedableRng;
use rayon::slice::ParallelSliceMut;
use roaring::RoaringBitmap;
use slice_group_by::GroupBy;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

fn parse_number_with_underscores(s: &str) -> Result<usize, std::num::ParseIntError> {
    s.replace('_', "").parse()
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The datasets to run and all of them are ran if empty.
    #[arg(long, value_enum, value_delimiter = ',')]
    datasets: Vec<scenarios::Dataset>,

    #[arg(long, value_enum, value_delimiter = ',')]
    contenders: Vec<scenarios::ScenarioContender>,

    #[arg(long, value_enum, value_delimiter = ',')]
    distances: Vec<scenarios::ScenarioDistance>,

    #[arg(long, value_enum, value_delimiter = ',')]
    over_samplings: Vec<scenarios::ScenarioOversampling>,

    #[arg(long, value_enum, value_delimiter = ',')]
    filterings: Vec<scenarios::ScenarioFiltering>,

    /// The list of recall to be tested.
    #[arg(long, value_delimiter = ',', default_value = "1,10,20,50,100,500")]
    recall_tested: Vec<usize>,

    /// Number of vectors to evaluate from the datasets.
    #[arg(long, default_value_t = 10_000, value_parser = parse_number_with_underscores)]
    count: usize,

    /// Set the number of trees to generate to a fixed value, if not specified the number of trees will be automatically computed.
    #[arg(long)]
    nb_trees: Option<usize>,

    /// These numbers correspond to the numbers of chunks that the dataset will be split into for indexing.
    ///
    /// Each number corresponds to a new indexation in x chunks. Use a comma to separate multiple features.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    number_of_chunks: Vec<usize>,

    /// The time to sleep between each chunk indexing specified in seconds.
    ///
    /// This is useful when profiling, it helps quickly identifying when each steps took place.
    /// Also, it's not counted in any of the individual reported indexing time metrics but it is counted in the total indexing time.
    #[arg(long, default_value_t = 0)]
    sleep_between_chunks: usize,

    /// Memory available for indexing.
    #[arg(long, default_value_t = Byte::MAX)]
    memory: Byte,

    /// The number of threads to use for indexing. If not specified the maximum number of threads will be used.
    #[arg(long)]
    threads: Option<usize>,

    /// When set to true, will print all the steps it goes through.
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() {
    let Args {
        datasets,
        count,
        nb_trees,
        number_of_chunks,
        contenders,
        distances,
        over_samplings,
        filterings,
        sleep_between_chunks,
        memory,
        recall_tested,
        threads,
        verbose,
    } = Args::parse();

    if verbose {
        // Initialize tracing with the specified level
        let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            let filter = format!("arroy=debug,benchmarks=debug");
            EnvFilter::new(filter)
        });

        FmtSubscriber::builder()
            .with_env_filter(env_filter)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .init();
    }

    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    }

    let datasets = set_or_all::<_, MatLEView<f32>>(datasets);
    let contenders = set_or_all::<_, scenarios::ScenarioContender>(contenders);
    let distances = set_or_all::<_, scenarios::ScenarioDistance>(distances);
    let over_samplings = set_or_all::<_, scenarios::ScenarioOversampling>(over_samplings);
    let filterings = set_or_all::<_, scenarios::ScenarioFiltering>(filterings);

    let scenaris: Vec<_> = iproduct!(datasets, distances, contenders, over_samplings, filterings)
        .map(|(dataset, distance, contender, oversampling, filtering)| {
            (dataset, distance, contender, ScenarioSearch { oversampling, filtering })
        })
        .sorted()
        .collect();

    let mut previous_dataset = None;
    for grp in scenaris
        .linear_group_by(|(da, dia, ca, _), (db, dib, cb, _)| da == db && dia == dib && ca == cb)
    {
        let (dataset, distance, contender, _) = &grp[0];
        let search: Vec<&ScenarioSearch> = grp.iter().map(|(_, _, _, s)| s).collect();

        if previous_dataset != Some(dataset.name()) {
            previous_dataset = Some(dataset.name());
            dataset.header();
            if dataset.len() != count {
                let c = count.min(dataset.len());
                println!(
                    "\x1b[1m{c}\x1b[0m vectors are used for this measure and {memory}B of memory",
                );
            }
        }

        let points: Vec<_> =
            dataset.iter().take(count).enumerate().map(|(i, v)| (i as u32, v)).collect();
        let memory = memory.as_u64() as usize;

        let mut recall_tested_s = String::new();
        recall_tested
            .iter()
            .for_each(|recall| write!(&mut recall_tested_s, "{recall:4}, ").unwrap());
        let recall_tested_s = recall_tested_s.trim_end_matches(", ");
        println!("Recall tested is:   [{recall_tested_s}]");

        let max = recall_tested.iter().max().copied().unwrap_or_default();
        // If we have no recall we can skip entirely the generation of the queries
        let queries = if max == 0 {
            Vec::new()
        } else {
            let mut rng = StdRng::seed_from_u64(RNG_SEED);
            (0..100)
                .map(|_| points.choose(&mut rng).unwrap())
                .map(|(id, target)| {
                    let mut points = points.clone();
                    points.par_sort_unstable_by_key(|(_, v)| {
                        OrderedFloat(benchmarks::distance::<Cosine>(target, v))
                    });

                    // We collect the different filtered versions here.
                    let filtered: HashMap<_, _> = search
                        .iter()
                        .map(|ScenarioSearch { filtering, .. }| {
                            let candidates = match filtering {
                                scenarios::ScenarioFiltering::NoFilter => None,
                                filtering => {
                                    let total = points.len() as f32;
                                    let filtering = filtering.to_ratio_f32();
                                    Some(
                                        points
                                            .iter()
                                            .map(|(id, _)| id)
                                            .take((total * filtering) as usize)
                                            .collect::<RoaringBitmap>(),
                                    )
                                }
                            };

                            // This is the real expected answer without the filtered out candidates.
                            let answer = points
                                .iter()
                                .map(|(id, _)| *id)
                                .filter(|&id| candidates.as_ref().map_or(true, |c| c.contains(id)))
                                .take(max)
                                .collect::<Vec<_>>();

                            (*filtering, (candidates, answer))
                        })
                        .collect();

                    (id, target, filtered)
                })
                .collect()
        };
        println!("Starting indexing process");

        for number_of_chunks in &number_of_chunks {
            match contender {
                scenarios::ScenarioContender::Qdrant => println!("Qdrant is not supported yet"),
                scenarios::ScenarioContender::Arroy => match distance {
                    scenarios::ScenarioDistance::Cosine => {
                        arroy_bench::prepare_and_run::<Cosine, _>(
                            &points,
                            nb_trees,
                            *number_of_chunks,
                            sleep_between_chunks,
                            memory,
                            verbose,
                            |time_to_index, env, database| {
                                arroy_bench::run_scenarios(
                                    env,
                                    time_to_index,
                                    distance,
                                    *number_of_chunks,
                                    &search,
                                    &queries,
                                    &recall_tested,
                                    database,
                                );
                            },
                        )
                    }
                },
            }
        }

        println!();
    }
}

fn set_or_all<S, T>(datasets: Vec<S>) -> Vec<T>
where
    S: Sequence,
    S: Into<T>,
{
    if datasets.is_empty() {
        enum_iterator::all::<S>().map(Into::into).collect()
    } else {
        datasets.into_iter().map(Into::into).collect()
    }
}
