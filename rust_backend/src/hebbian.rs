//! Supervised Hebbian learning: zero initialization, one ordered pass, then verification.

use super::{activate, validate as validate_adaline, Epoch, Model, Options, Sample};

#[derive(Debug)]
pub struct Step {
    pub epoch: usize,
    pub pattern: usize,
    pub before: Model,
    pub delta_weights: Vec<f64>,
    pub delta_bias: f64,
    pub after: Model,
    pub max_change: f64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Status {
    PassComplete,
    NumericalLimit,
}

impl Status {
    pub fn name(&self) -> &'static str {
        match self {
            Self::PassComplete => "pass_complete",
            Self::NumericalLimit => "numerical_limit",
        }
    }
}

#[derive(Debug)]
pub struct Training {
    pub status: Status,
    pub failed_at: Option<(usize, usize)>,
    pub steps: Vec<Step>,
    pub epochs: Vec<Epoch>,
    pub model: Model,
}

pub fn diagnostics(samples: &[Sample], model: &Model) -> (f64, usize) {
    let mut sse = 0.0;
    let mut correct = 0;
    for sample in samples {
        let output = f64::from(activate(model.net(&sample.inputs)));
        sse += (sample.target - output).powi(2);
        correct += usize::from(output == sample.target);
    }
    (sse, correct)
}

pub fn train(samples: &[Sample], eta: f64) -> Result<Training, String> {
    let n = samples
        .first()
        .ok_or("Enter at least one training pattern.")?
        .inputs
        .len();
    // Step 1: zero initialization is part of this algorithm, not a caller option.
    let initial = Model {
        weights: vec![0.0; n],
        bias: 0.0,
    };
    validate_adaline(
        samples,
        &initial,
        Options {
            alpha: eta,
            tolerance: 1.0,
            max_epochs: 1,
        },
    )?;
    let mut result = Training {
        status: Status::PassComplete,
        failed_at: None,
        steps: Vec::new(),
        epochs: Vec::new(),
        model: initial,
    };
    let mut max_change: f64 = 0.0;
    // Step 2: visit every pattern exactly once, in the supplied order.
    for (index, sample) in samples.iter().enumerate() {
        let before = result.model.clone();
        // Step 3: use target t directly; no prediction or error gates the update.
        let delta_weights: Vec<f64> = sample
            .inputs
            .iter()
            .map(|&x| eta * x * sample.target)
            .collect();
        let delta_bias = eta * sample.target;
        // Step 4: commit this pattern before processing the next one.
        let after = Model {
            weights: before
                .weights
                .iter()
                .zip(&delta_weights)
                .map(|(w, d)| w + d)
                .collect(),
            bias: before.bias + delta_bias,
        };
        if delta_weights
            .iter()
            .chain([&delta_bias, &after.bias])
            .chain(after.weights.iter())
            .any(|v| !v.is_finite() || v.abs() > 1e100)
        {
            result.status = Status::NumericalLimit;
            result.failed_at = Some((1, index + 1));
            return Ok(result);
        }
        let change = delta_weights
            .iter()
            .map(|v| v.abs())
            .fold(delta_bias.abs(), f64::max);
        max_change = max_change.max(change);
        result.model = after.clone();
        result.steps.push(Step {
            epoch: 1,
            pattern: index + 1,
            before,
            delta_weights,
            delta_bias,
            after,
            max_change: change,
        });
    }
    // Step 5: verify all patterns with ONE fixed model, after the entire pass.
    let (sse, correct) = diagnostics(samples, &result.model);
    let net_change = result
        .model
        .weights
        .iter()
        .map(|v| v.abs())
        .fold(result.model.bias.abs(), f64::max);
    result.epochs.push(Epoch {
        number: 1,
        max_change,
        net_change,
        sse,
        correct,
        model: result.model.clone(),
    });
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn and() -> Vec<Sample> {
        vec![
            (vec![1., 1.], 1.),
            (vec![1., -1.], -1.),
            (vec![-1., 1.], -1.),
            (vec![-1., -1.], -1.),
        ]
        .into_iter()
        .map(|(inputs, target)| Sample { inputs, target })
        .collect()
    }

    #[test]
    fn hand_worked_and_starts_at_zero_and_updates_every_pattern_once() {
        let r = train(&and(), 1.).unwrap();
        assert_eq!(r.status, Status::PassComplete);
        assert_eq!(r.steps.len(), 4);
        assert_eq!(r.epochs.len(), 1);
        let expected = [
            (vec![1., 1.], 1.),
            (vec![0., 2.], 0.),
            (vec![1., 1.], -1.),
            (vec![2., 2.], -2.),
        ];
        for (s, (weights, bias)) in r.steps.iter().zip(expected) {
            assert_eq!(s.after.weights, weights);
            assert_eq!(s.after.bias, bias);
        }
        // At net zero the first pattern already predicts +1, but still updates.
        assert_eq!(r.steps[0].before.weights, vec![0., 0.]);
        assert_eq!(r.steps[0].delta_weights, vec![1., 1.]);
        assert_eq!(r.epochs[0].correct, 4);
        assert_eq!(r.epochs[0].sse, 0.);
    }

    #[test]
    fn pass_completion_does_not_claim_correctness_or_repeat_nonseparable_data() {
        let mut samples = and();
        for (sample, target) in samples.iter_mut().zip([-1., 1., 1., -1.]) {
            sample.target = target;
        }
        let r = train(&samples, 1.).unwrap();
        assert_eq!(r.status, Status::PassComplete);
        assert_eq!(r.steps.len(), 4);
        assert_eq!(r.model.weights, vec![0., 0.]);
        assert_eq!(r.model.bias, 0.);
        assert_eq!(r.epochs[0].correct, 2);
        assert_eq!(r.epochs[0].sse, 8.);
    }

    #[test]
    fn three_input_deltas_use_target_and_eta_not_a_prediction_error() {
        let samples = vec![Sample {
            inputs: vec![2., -3., 0.5],
            target: -1.,
        }];
        let r = train(&samples, 0.25).unwrap();
        assert_eq!(r.steps[0].delta_weights, vec![-0.5, 0.75, -0.125]);
        assert_eq!(r.model.bias, -0.25);
    }

    #[test]
    fn preserves_row_order_and_reports_an_incomplete_pass_on_numerical_limit() {
        let mut samples = and();
        samples.reverse();
        let r = train(&samples, 1.).unwrap();
        assert_eq!(r.steps[0].delta_weights, vec![1., 1.]);
        assert_eq!(r.steps[0].delta_bias, -1.);
        assert_eq!(r.model.weights, vec![2., 2.]);
        let large = 2.0_f64.powi(165);
        let samples = vec![
            Sample {
                inputs: vec![large],
                target: 1.
            };
            5
        ];
        let r = train(&samples, large).unwrap();
        assert_eq!(r.status, Status::NumericalLimit);
        assert_eq!(r.failed_at, Some((1, 5)));
        assert_eq!(r.steps.len(), 4);
        assert!(r.epochs.is_empty());
        assert_eq!(r.model.weights, vec![2.0_f64.powi(332)]);
    }

    #[test]
    fn invalid_rate_targets_shapes_and_empty_data_are_rejected() {
        for eta in [0., -1., f64::NAN, f64::INFINITY] {
            assert!(train(&and(), eta).is_err());
        }
        assert!(train(&[], 1.).is_err());
        assert!(train(
            &[Sample {
                inputs: vec![1.],
                target: 0.
            }],
            1.
        )
        .is_err());
        let mut samples = and();
        samples[1].inputs.pop();
        assert!(train(&samples, 1.).is_err());
    }
}
