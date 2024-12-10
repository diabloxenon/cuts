/*
 Online Welch's t-test

 Tests whether two populations have same mean.
 This is basically Student's t-test for unequal
 variances and unequal sample sizes.

 see https://en.wikipedia.org/wiki/Welch%27s_t-test
*/

use crate::CUTS_ENOUGH_MEASUREMENTS;

// threshold values for Welch's t-test
pub(crate) const T_THRESHOLD_BANANAS: f64 = 500.0; // test failed, with overwhelming probability
pub(crate) const T_THRESHOLD_MODERATE: f64 = 10.0; // test failed. 

#[derive(Clone, Copy)]
pub struct CutsTTest {
    pub mean: [f64; 2],
    pub m2: [f64; 2],
    pub n: [f64; 2],
}

impl Default for CutsTTest {
    fn default() -> Self {
        Self {
            mean: [0.0, 0.0],
            m2: [0.0, 0.0],
            n: [0.0, 0.0],
        }
    }
}

impl CutsTTest {
    pub(crate) fn init() -> Self {
        CutsTTest::default()
    }

    pub(crate) fn compute(&self) -> f64 {
        let mut var: [f64; 2] = [0.0, 0.0];
        var[0] = self.m2[0] / (self.n[0] - 1.0);
        var[1] = self.m2[1] / (self.n[1] - 1.0);
        let num = self.mean[0] - self.mean[1];
        let den = (var[0] / self.n[0] + var[1] / self.n[1]).sqrt();
        let t_value = num / den;
        t_value
    }

    pub(crate) fn push(&mut self, x: f64, clazz: usize) {
        assert!(clazz == 0 || clazz == 1);
        self.n[clazz] += 1.0;
        /*
         estimate variance on the fly as per the Welford method.
         this gives good numerical stability, see Knuth's TAOCP vol 2
        */
        let delta: f64 = x - self.mean[clazz];
        self.mean[clazz] = self.mean[clazz] + delta / self.n[clazz];
        self.m2[clazz] = self.m2[clazz] + delta * (x - self.mean[clazz]);
    }

    pub(crate) fn report_test(&self) {
        if self.n[0] > CUTS_ENOUGH_MEASUREMENTS {
            let tval = self.compute();
            println!(
                "abs(t): {}, number measurements: {}",
                tval,
                self.n[0] + self.n[1]
            )
        } else {
            println!("not enough measurements: {} + {}", self.n[0], self.n[1])
        }
    }
}
