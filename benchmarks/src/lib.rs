#![allow(clippy::type_complexity)]

pub mod arroy_bench;
mod dataset;
pub mod hannoy_bench;
mod qdrant_bench;
pub mod scenarios;

use std::fmt;
use std::time::Instant;

use byte_unit::rust_decimal::Decimal;
use byte_unit::{Byte, Unit, UnitType};
pub use dataset::*;
use qdrant_client::qdrant::quantization_config;

pub const RNG_SEED: u64 = 38;

/// A generalist distance trait that contains the informations required to configure every engine
pub trait Distance {
    const BINARY_QUANTIZED: bool;
    const QDRANT_DISTANCE: qdrant_client::qdrant::Distance;
    type ArroyDistance: arroy::Distance;
    type HannoyDistance: hannoy::Distance;

    fn name() -> &'static str;
    fn qdrant_quantization_config() -> quantization_config::Quantization;
    fn real_distance(a: &[f32], b: &[f32]) -> f32;
}

macro_rules! arroy_distance {
    ($distance:ident => real: $real:ident, qdrant: $qdrant:ident, bq: $bq:expr) => {
        pub struct $distance {}

        impl Distance for $distance {
            const BINARY_QUANTIZED: bool = $bq;
            const QDRANT_DISTANCE: qdrant_client::qdrant::Distance =
                qdrant_client::qdrant::Distance::$qdrant;
            type ArroyDistance = arroy::distances::$distance;
            type HannoyDistance = hannoy::distances::$distance;

            fn name() -> &'static str {
                stringify!($distance)
            }
            fn qdrant_quantization_config() -> quantization_config::Quantization {
                qdrant_client::qdrant::BinaryQuantization::default().into()
            }
            fn real_distance(a: &[f32], b: &[f32]) -> f32 {
                let a = ndarray::aview1(a);
                let b = ndarray::aview1(b);
                fast_distances::$real(&a, &b)
            }
        }
    };
}

arroy_distance!(BinaryQuantizedCosine => real: cosine, qdrant: Cosine, bq: true);
arroy_distance!(Cosine =>  real: cosine, qdrant: Cosine, bq: false);
arroy_distance!(BinaryQuantizedEuclidean => real: euclidean, qdrant: Euclid, bq: true);
arroy_distance!(Euclidean => real: euclidean, qdrant: Euclid, bq: false);
arroy_distance!(BinaryQuantizedManhattan => real: manhattan, qdrant: Manhattan, bq: true);
arroy_distance!(Manhattan => real: manhattan, qdrant: Manhattan, bq: false);
// arroy_distance!(DotProduct => real: dot, qdrant: Dot);

pub fn distance<D: crate::Distance>(left: &[f32], right: &[f32]) -> f32 {
    D::real_distance(left, right)
}

pub struct Recall(pub f32);

impl fmt::Debug for Recall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            // red
            f32::NEG_INFINITY..=0.25 => write!(f, "\x1b[1;31m")?,
            // yellow
            0.25..=0.5 => write!(f, "\x1b[1;33m")?,
            // green
            0.5..=0.75 => write!(f, "\x1b[1;32m")?,
            // blue
            0.75..=0.90 => write!(f, "\x1b[1;34m")?,
            // cyan
            0.90..=0.999 => write!(f, "\x1b[1;36m")?,
            // underlined cyan
            0.999..=f32::INFINITY => write!(f, "\x1b[1;4;36m")?,
            _ => (),
        }
        write!(f, "{:.2}\x1b[0m", self.0)
    }
}

#[derive(Debug)]
pub struct IndexingMetrics {
    start: Instant,
    end: Instant,
    insert_durations: Vec<(Instant, Instant)>,
    build_durations: Vec<(Instant, Instant)>,
    nb_vectors: Vec<usize>,
    database_size: Vec<usize>,
    nb_trees: Vec<usize>,
}

impl IndexingMetrics {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            end: Instant::now(),
            insert_durations: Vec::new(),
            build_durations: Vec::new(),
            nb_vectors: Vec::new(),
            database_size: Vec::new(),
            nb_trees: Vec::new(),
        }
    }

    pub fn start_insertion(&mut self) {
        self.insert_durations.push((Instant::now(), Instant::now()));
    }

    pub fn end_insertion(&mut self) {
        self.insert_durations.last_mut().unwrap().1 = Instant::now();
    }

    pub fn start_building(&mut self) {
        self.build_durations.push((Instant::now(), Instant::now()));
    }

    pub fn end_building(&mut self) {
        self.build_durations.last_mut().unwrap().1 = Instant::now();
    }

    pub fn new_nb_vectors(&mut self, nb_vectors: usize) {
        self.nb_vectors.push(nb_vectors);
    }

    pub fn new_database_size(&mut self, size: usize) {
        self.database_size.push(size);
    }
    pub fn new_nb_trees(&mut self, nb_trees: usize) {
        self.nb_trees.push(nb_trees);
    }

    pub fn end(&mut self) {
        self.end = Instant::now();
    }
}

impl fmt::Display for IndexingMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Total time to index: {:.2?} (", self.end.duration_since(self.start))?;

        for (idx, ((insert_start, insert_end), (build_start, build_end))) in
            self.insert_durations.iter().zip(self.build_durations.iter()).enumerate()
        {
            if idx != 0 {
                write!(f, " + ")?;
            }
            write!(
                f,
                "{:.2?}",
                insert_end.duration_since(*insert_start) + build_end.duration_since(*build_start)
            )?;
        }
        writeln!(f, ")")?;

        // First step is to format all the lists in a vector of strings

        let vectors = self.nb_vectors.iter().map(|v| format!("{v}")).collect::<Vec<_>>();
        let insertions = self
            .insert_durations
            .iter()
            .map(|(insert_start, insert_end)| {
                format!("{:.2?}", insert_end.duration_since(*insert_start))
            })
            .collect::<Vec<_>>();
        let builds = self
            .build_durations
            .iter()
            .map(|(build_start, build_end)| {
                format!("{:.2?}", build_end.duration_since(*build_start))
            })
            .collect::<Vec<_>>();
        let trees = self.nb_trees.iter().map(|v| format!("{v}")).collect::<Vec<_>>();
        let db_size = self
            .database_size
            .iter()
            .map(|v| {
                format!(
                    "{:.2}",
                    Byte::from_decimal_with_unit(Decimal::from(*v), Unit::B)
                        .unwrap()
                        .get_appropriate_unit(UnitType::Binary)
                )
            })
            .collect::<Vec<_>>();

        // Then we can retrieve the max length of each column in the list to pretty print the table later
        let max_lengths = vectors
            .iter()
            .zip(insertions.iter())
            .zip(builds.iter())
            .zip(trees.iter())
            .zip(db_size.iter())
            .map(|((((v, i), b), t), d)| {
                [v.len(), i.len(), b.len(), t.len(), d.len()].into_iter().max().unwrap()
            })
            .collect::<Vec<_>>();

        write!(f, "  => Vectors:    ")?;
        for (idx, (nb_vectors, max_length)) in vectors.iter().zip(max_lengths.iter()).enumerate() {
            if idx != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{nb_vectors:>max_length$}")?;
        }
        writeln!(f)?;

        write!(f, "  => Insertions: ")?;
        for (idx, (insert, max_length)) in insertions.iter().zip(max_lengths.iter()).enumerate() {
            if idx != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{insert:>max_length$}")?;
        }
        writeln!(f)?;

        write!(f, "  => Builds:     ")?;
        for (idx, (build, max_length)) in builds.iter().zip(max_lengths.iter()).enumerate() {
            if idx != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{build:>max_length$}")?;
        }
        writeln!(f)?;

        write!(f, "  => Trees:      ")?;
        for (idx, (nb_trees, max_length)) in trees.iter().zip(max_lengths.iter()).enumerate() {
            if idx != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{nb_trees:>max_length$}")?;
        }
        writeln!(f)?;

        write!(f, "  => Db size:    ")?;
        for (idx, (database_size, max_length)) in db_size.iter().zip(max_lengths.iter()).enumerate()
        {
            if idx != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{database_size:>max_length$}")?;
        }

        Ok(())
    }
}
