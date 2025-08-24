use std::collections::HashMap;
use std::time::Duration;

use byte_unit::{Byte, UnitType};
use hannoy::internals::{self, NodeCodec};
use hannoy::{Database, Distance, ItemId, Writer};
use heed::EnvOpenOptions;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use roaring::RoaringBitmap;

use crate::Recall;
use crate::{scenarios::*, IndexingMetrics};
const TWENTY_HUNDRED_MIB: usize = 2000 * 1024 * 1024 * 1024;

pub fn prepare_and_run<D, F>(
    points: &[(u32, &[f32])],
    number_of_chunks: usize,
    sleep_between_chunks: usize,
    memory: usize,
    verbose: bool,
    execute: F,
) where
    D: Distance,
    F: FnOnce(&IndexingMetrics, &heed::Env, Database<D>),
{
    let dimensions = points[0].1.len();

    let dir = tempfile::tempdir().unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path()) }.unwrap();

    let mut hannoy_seed = StdRng::seed_from_u64(13);
    let mut wtxn = env.write_txn().unwrap();
    let database =
        env.create_database::<internals::KeyCodec, NodeCodec<D>>(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();

    let duration = load_into_hannoy(
        &mut hannoy_seed,
        &env,
        database,
        dimensions,
        memory,
        points,
        number_of_chunks,
        sleep_between_chunks,
        verbose,
    );

    (execute)(&duration, &env, database);
}

#[allow(clippy::too_many_arguments)]
pub fn run_scenarios<D: Distance>(
    env: &heed::Env,
    time_to_index: &IndexingMetrics,
    distance: &ScenarioDistance,
    number_of_chunks: usize,
    search: &[&ScenarioSearch],
    queries: &[(&u32, &&[f32], HashMap<ScenarioFiltering, (Option<RoaringBitmap>, Vec<u32>)>)],
    recall_tested: &[usize],
    database: hannoy::Database<D>,
) {
    let database_size =
        Byte::from_u64(env.non_free_pages_size().unwrap()).get_appropriate_unit(UnitType::Binary);

    println!("Database size: {database_size:#.2}, indexed in {number_of_chunks} chunks");
    println!("{time_to_index}");

    for ScenarioSearch { oversampling, filtering } in search {
        let mut time_to_search = Duration::default();
        let mut recalls = Vec::new();
        for &number_fetched in recall_tested {
            let (correctly_retrieved, duration) = queries
                .par_iter()
                .map(|(&id, _target, relevants)| {
                    let rtxn = env.read_txn().unwrap();
                    let reader = hannoy::Reader::open(&rtxn, 0, database).unwrap();

                    let (candidates, relevants) = &relevants[filtering];
                    // Only keep the top number fetched documents.
                    let relevants = relevants.get(..number_fetched).unwrap_or(relevants);

                    let now = std::time::Instant::now();
                    let mut nns = reader.nns(number_fetched);
                    if let Some(candidates) = candidates.as_ref() {
                        nns.candidates(candidates);
                    }
                    let arroy_answer = nns.by_item(&rtxn, id).unwrap().unwrap();
                    let elapsed = now.elapsed();

                    let mut correctly_retrieved = Some(0);
                    for (id, _dist) in arroy_answer {
                        if relevants.contains(&id) {
                            if let Some(cr) = &mut correctly_retrieved {
                                *cr += 1;
                            }
                        } else if let Some(cand) = candidates.as_ref() {
                            // We set the counter to -1 if we return a filtered out candidated
                            if !cand.contains(id) {
                                correctly_retrieved = None;
                            }
                        }
                    }

                    (correctly_retrieved, elapsed)
                })
                .reduce(
                    || (Some(0), Duration::default()),
                    |(aanswer, aduration), (banswer, bduration)| {
                        (aanswer.zip(banswer).map(|(a, b)| a + b), aduration + bduration)
                    },
                );

            time_to_search += duration;
            // If non-candidate documents are returned we show a recall of -1
            let recall =
                correctly_retrieved.map_or(-1.0, |cr| cr as f32 / (number_fetched as f32 * 100.0));
            recalls.push(Recall(recall));
        }

        let filtered_percentage = filtering.to_ratio_f32() * 100.0;
        println!(
            "[hannoy] {distance:16?} {oversampling}: {recalls:?}, \
                                    searched for: {time_to_search:02.2?}, \
                                    searched in {filtered_percentage:#.2}%"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn load_into_hannoy<D: hannoy::Distance>(
    rng: &mut StdRng,
    env: &heed::Env,
    database: Database<D>,
    dimensions: usize,
    memory: usize,
    points: &[(ItemId, &[f32])],
    number_of_chunks: usize,
    sleep_between_chunks: usize,
    verbose: bool,
) -> IndexingMetrics {
    let mut metrics = IndexingMetrics::new();
    let avg_chunk_size = points.len() / number_of_chunks;
    let mut nb_vectors = 0;

    for points in points.chunks(avg_chunk_size) {
        if sleep_between_chunks != 0 {
            std::thread::sleep(Duration::from_secs(sleep_between_chunks as u64));
        }
        tracing::info!("Inserting chunk of size {} in arroy", points.len());
        let mut wtxn = env.write_txn().unwrap();
        metrics.start_insertion();
        let writer = Writer::<D>::new(database, 0, dimensions);
        for (i, vector) in points.iter() {
            assert_eq!(vector.len(), dimensions);
            writer.add_item(&mut wtxn, *i, vector).unwrap();
        }
        metrics.end_insertion();

        tracing::info!("Starts building the trees");

        let mut builder = writer.builder(rng);
        builder.available_memory(memory);
        let progress = steppe::default::DefaultProgress::default();
        metrics.start_building();
        if verbose {
            progress.follow_progression_on_tty();
            builder.progress(progress.clone()).build::<16, 32>(&mut wtxn).unwrap();
        } else {
            builder.build::<16, 32>(&mut wtxn).unwrap();
        }
        metrics.end_building();
        progress.finish();
        wtxn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();
        let reader = hannoy::Reader::open(&rtxn, 0, database).unwrap();
        metrics.new_nb_trees(reader.n_trees());
        drop(rtxn);

        nb_vectors += points.len();
        metrics.new_nb_vectors(nb_vectors);
        metrics.new_database_size(env.non_free_pages_size().unwrap() as usize);
    }

    metrics.end();
    metrics
}
