use std::{
    arch::x86_64::{_mm_mfence, _rdtsc},
    cmp::Ordering,
};

use rand::{Fill, Rng};
use ttest::{CutsTTest, T_THRESHOLD_BANANAS, T_THRESHOLD_MODERATE};

mod ttest;

pub const CUTS_ENOUGH_MEASUREMENTS: f64 = 10000.0;
pub const CUTS_NUMBER_PERCENTILES: usize = 100;

pub const CUTS_TESTS: usize = 1 + CUTS_NUMBER_PERCENTILES + 1;

pub struct CutsConfig {
    chunk_size: usize,
    number_measurements: usize,
}

pub struct CutsContext {
    ticks: Vec<u64>,
    exec_times: Vec<f64>,
    input_data: Vec<u8>,
    classes: Vec<usize>,
    config: CutsConfig,
    ttest_ctx: [CutsTTest; CUTS_TESTS],
    percentiles: [f64; CUTS_NUMBER_PERCENTILES],
}

#[derive(PartialEq)]
pub enum CutsState {
    LeakageFound,
    NoLeakageEvidenceYet,
}

/*
 Returns current CPU tick count from *T*ime *S*tamp *C*ounter.

 To enforce CPU to issue RDTSC instruction where we want it to, we put a `mfence` instruction before
 issuing `rdtsc`, which should make all memory load/ store operations, prior to RDTSC, globally visible.

 See https://github.com/oreparaz/dudect/issues/32
 See RDTSC documentation @ https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.htm#text=rdtsc&ig_expand=4395,5273
 See MFENCE documentation @ https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.htm#text=mfence&ig_expand=4395,5273,4395

 Also see https://stackoverflow.com/a/12634857
*/
#[inline]
fn cpucycles() -> u64 {
    unsafe {
        _mm_mfence();
        _rdtsc()
    }
}

// Helper function: extract a value representing the `pct` percentile of a sorted sample-set, using
// linear interpolation. If samples are not sorted, return nonsensical value.
fn percentile_of_sorted(sorted_samples: &[f64], pct: f64) -> f64 {
    assert!(!sorted_samples.is_empty());
    if sorted_samples.len() == 1 {
        return sorted_samples[0];
    }
    let zero: f64 = 0.0;
    assert!(zero <= pct);
    let hundred = 100_f64;
    assert!(pct <= hundred);
    if pct == hundred {
        return sorted_samples[sorted_samples.len() - 1];
    }
    let length = (sorted_samples.len() - 1) as f64;
    let rank = (pct / hundred) * length;
    let lrank = rank.floor();
    let d = rank - lrank;
    let n = lrank as usize;
    let lo = sorted_samples[n];
    let hi = sorted_samples[n + 1];
    lo + (hi - lo) * d
}

impl CutsContext {
    // /*
    //  set different thresholds for cropping measurements.
    //  the exponential tendency is meant to approximately match
    //  the measurements distribution, but there's not more science
    //  than that.
    // */
    fn prepare_percentiles(&mut self) {
        self.exec_times.sort_by(|a, b| a.total_cmp(b));
        for i in 0..CUTS_NUMBER_PERCENTILES {
            let pct = 0.5_f64.powf(10.0 * (i + 1) as f64 / CUTS_NUMBER_PERCENTILES as f64);
            self.percentiles[i] = percentile_of_sorted(self.exec_times.as_ref(), pct);
        }
    }

    fn measure<F>(&mut self, do_one_computation: F)
    where
        F: Fn(&[u8]) -> u8,
    {
        for i in 0..self.config.number_measurements {
            self.ticks[i] = cpucycles();
            do_one_computation(&self.input_data[i..i * self.config.chunk_size]);
        }

        for i in 0..self.config.number_measurements - 1 {
            self.exec_times[i] = self.ticks[i + 1] as f64 - self.ticks[i] as f64;
        }
    }

    fn update_statistics(&mut self) {
        for i in 10..self.config.number_measurements - 1 {
            let difference = self.exec_times[i] as i64;
            if difference < 0 {
                continue; // the cpu cycle counter overflowed, just throw away the measurement
            }

            // t-test on the execution time
            self.ttest_ctx[0].push(difference as f64, self.classes[i]);

            // t-test on cropped execution times, for several cropping thresholds.
            for crop_idx in 0..CUTS_NUMBER_PERCENTILES {
                if difference < self.percentiles[crop_idx] as i64 {
                    self.ttest_ctx[crop_idx + 1].push(difference as f64, self.classes[i]);
                }
            }

            // second-order test (only if we have more than 10000 measurements).
            // Centered product pre-processing.
            if self.ttest_ctx[0].n[0] > 10000.0 {
                let centered: f64 = difference as f64 - self.ttest_ctx[0].mean[self.classes[i]];
                self.ttest_ctx[1 + CUTS_NUMBER_PERCENTILES]
                    .push(centered * centered, self.classes[i]);
            }
        }
    }

    fn max_test(&self) -> CutsTTest {
        let mut ret = 0;
        let mut max = 0_f64;

        for i in 0..CUTS_TESTS {
            if self.ttest_ctx[i].n[0] > CUTS_ENOUGH_MEASUREMENTS {
                let x = self.ttest_ctx[i].compute().abs();
                if max < x {
                    max = x;
                    ret = i;
                }
            }
        }
        self.ttest_ctx[ret]
    }

    fn report(&self) -> CutsState {
        for i in 0..CUTS_TESTS {
            println!(
                " bucket {} has {} measurements",
                i,
                self.ttest_ctx[i].n[0] + self.ttest_ctx[i].n[1]
            );
        }
        println!("t-test on raw measurements");
        self.ttest_ctx[0].report_test();

        println!("t-test on cropped measurements");
        for i in 0..CUTS_NUMBER_PERCENTILES {
            self.ttest_ctx[i + 1].report_test();
        }

        println!("t-test for second order leakage");
        self.ttest_ctx[1 + CUTS_NUMBER_PERCENTILES].report_test();

        let t: CutsTTest = self.max_test();
        let max_t = t.compute().abs();
        let number_traces_max_t = t.n[0] + t.n[1];
        let max_tau = max_t / number_traces_max_t.sqrt();

        // print the number of measurements of the test that yielded max t.
        // sometimes you can see this number go down - this can be confusing
        // but can happen (different test)

        println!("meas: {} M", number_traces_max_t / 1000000.0);
        if number_traces_max_t < CUTS_ENOUGH_MEASUREMENTS {
            println!(
                "not enough measurements ({} still to go).",
                CUTS_ENOUGH_MEASUREMENTS - number_traces_max_t
            );
            return CutsState::NoLeakageEvidenceYet;
        }
        /*
         * We report the following statistics:
         *
         * max_t: the t value
         * max_tau: a t value normalized by sqrt(number of measurements).
         *          this way we can compare max_tau taken with different
         *          number of measurements. This is sort of "distance
         *          between distributions", independent of number of
         *          measurements.
         * (5/tau)^2: how many measurements we would need to barely
         *            detect the leak, if present. "barely detect the
         *            leak" here means have a t value greater than 5.
         *
         * The first metric is standard; the other two aren't (but
         * pretty sensible imho)
         */
        println!(
            "max t: {}, max tau: {}, (5/tau)^2: {}.",
            max_t,
            max_tau,
            (5 * 5) as f64 / (max_tau * max_tau) as f64
        );

        if max_t > T_THRESHOLD_BANANAS {
            println!("Definitely not constant time.");
            return CutsState::LeakageFound;
        }
        if max_t > T_THRESHOLD_MODERATE {
            println!("Probably not constant time.");
            return CutsState::LeakageFound;
        }
        if max_t < T_THRESHOLD_MODERATE {
            println!(" For the moment, maybe constant time.");
        }
        CutsState::NoLeakageEvidenceYet
    }

    pub fn run<F, G>(&mut self, prepare_inputs: G, do_one_computation: F) -> CutsState
    where
        F: Fn(&[u8]) -> u8,
        G: Fn(&CutsConfig, &mut [usize]) -> Vec<u8>,
    {
        self.input_data = prepare_inputs(&self.config, &mut self.classes);
        self.measure(do_one_computation);
        let first_time = self.percentiles[CUTS_NUMBER_PERCENTILES - 1] == 0.0;
        let mut ret: CutsState = CutsState::NoLeakageEvidenceYet;
        if first_time {
            // throw away the first batch of measurements.
            // this helps warming things up.
            self.prepare_percentiles();
        } else {
            self.update_statistics();
            ret = self.report();
        }
        ret
    }

    pub fn init(config: CutsConfig) -> Self {
        Self {
            ticks: vec![0; config.number_measurements],
            exec_times: vec![0.0; config.number_measurements],
            input_data: vec![0; config.number_measurements * config.chunk_size],
            classes: vec![0; config.number_measurements],
            config,
            ttest_ctx: [CutsTTest::default(); CUTS_TESTS],
            percentiles: [0.0; CUTS_NUMBER_PERCENTILES],
        }
    }
}

//##############################

const SECRET_LEN_BYTES: usize = 8;

const SECRET: [u8; SECRET_LEN_BYTES] = [0, 1, 2, 3, 4, 5, 6, 42];

//Target function to check for constant time
fn check_tag(x: &[u8], y: &[u8]) -> Ordering {
    x.cmp(y)
}

fn do_one_computation(data: &[u8]) -> u8 {
    check_tag(data, &SECRET);
    0
}

fn prepare_inputs(config: &CutsConfig, classes: &mut [usize]) -> Vec<u8> {
    let mut input_data: Vec<u8> = (0..config.chunk_size * config.number_measurements)
        .map(|_| rand::random::<u8>())
        .collect();
    for i in 0..config.number_measurements {
        classes[i] = rand::thread_rng().gen_range(0..=1);
        if classes[i] == 0 {
            input_data[i..i * config.chunk_size].fill(0x00);
        } else {
            // leave random
        }
    }
    input_data
}

fn main() {
    let config = CutsConfig {
        chunk_size: SECRET_LEN_BYTES,
        number_measurements: 500,
    };
    let mut context = CutsContext::init(config);

    let mut state = CutsState::NoLeakageEvidenceYet;
    while state == CutsState::NoLeakageEvidenceYet {
        state = context.run(prepare_inputs, do_one_computation);
    }
}
