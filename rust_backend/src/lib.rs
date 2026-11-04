//! Online ADALINE, steps 0–6 of the supplied worksheet.
//! Thresholding is for display/prediction only; learning always uses linear net input.

pub mod hebbian;
pub mod madaline;

pub const MAX_STEPS: usize = 100_000;
pub const MAX_COORDINATES: usize = 1_000_000;

#[derive(Clone, Debug)]
pub struct Sample {
    pub inputs: Vec<f64>,
    pub target: f64,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Model {
    pub fn net(&self, inputs: &[f64]) -> f64 {
        self.bias
            + self
                .weights
                .iter()
                .zip(inputs)
                .map(|(w, x)| w * x)
                .sum::<f64>()
    }
}

pub fn activate(net: f64) -> i8 {
    if net >= 0.0 {
        1
    } else {
        -1
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub alpha: f64,
    pub tolerance: f64,
    pub max_epochs: usize,
}

#[derive(Debug)]
pub struct Step {
    pub epoch: usize,
    pub pattern: usize,
    pub before: Model,
    pub net: f64,
    pub output: i8,
    pub error: f64,
    pub delta_weights: Vec<f64>,
    pub delta_bias: f64,
    pub after: Model,
    pub max_change: f64,
}

#[derive(Debug)]
pub struct Epoch {
    pub number: usize,
    pub max_change: f64,
    pub net_change: f64,
    pub sse: f64,
    pub correct: usize,
    pub model: Model,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Status {
    ToleranceMet,
    EpochLimit,
    NumericalLimit,
}

impl Status {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ToleranceMet => "tolerance_met",
            Self::EpochLimit => "epoch_limit",
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

fn bounded(value: f64) -> bool {
    value.is_finite() && value.abs() <= 1e50
}

pub fn validate(samples: &[Sample], model: &Model, options: Options) -> Result<(), String> {
    if !bounded(options.alpha) || options.alpha <= 0.0 {
        return Err("Learning rate must be positive, finite, and at most 1e50.".into());
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err("Tolerance must be positive and finite.".into());
    }
    let n = model.weights.len();
    if !(1..=32).contains(&n) {
        return Err("Use 1 to 32 input coordinates.".into());
    }
    if samples.is_empty() || samples.len() > 128 {
        return Err("Use 1 to 128 training patterns.".into());
    }
    if options.max_epochs == 0 || options.max_epochs > 100_000 {
        return Err("Maximum epochs must be between 1 and 100,000.".into());
    }
    let steps = samples.len() * options.max_epochs;
    if steps > MAX_STEPS || steps * n > MAX_COORDINATES {
        return Err("Reduce the epoch limit or dataset: the full trace supports 100,000 updates and 1,000,000 input coordinates.".into());
    }
    if !bounded(model.bias) || model.weights.iter().any(|&v| !bounded(v)) {
        return Err(
            "Initial weights and bias must be finite and at most 1e50 in magnitude.".into(),
        );
    }
    for sample in samples {
        if sample.inputs.len() != n || sample.inputs.iter().any(|&v| !bounded(v)) {
            return Err("Every pattern must match the weights and have finite inputs of magnitude at most 1e50.".into());
        }
        if sample.target != -1.0 && sample.target != 1.0 {
            return Err("Targets must be bipolar: -1 or +1.".into());
        }
    }
    Ok(())
}

pub fn diagnostics(samples: &[Sample], model: &Model) -> (f64, usize) {
    let mut sse = 0.0;
    let mut correct = 0;
    for sample in samples {
        let net = model.net(&sample.inputs);
        sse += (sample.target - net).powi(2);
        correct += usize::from(f64::from(activate(net)) == sample.target);
    }
    (sse, correct)
}

pub fn train(samples: &[Sample], initial: Model, options: Options) -> Result<Training, String> {
    validate(samples, &initial, options)?;
    let mut result = Training {
        status: Status::EpochLimit,
        failed_at: None,
        steps: Vec::new(),
        epochs: Vec::new(),
        model: initial,
    };
    // Steps 1–2: fixed row order, one update for EVERY sample.
    for epoch in 1..=options.max_epochs {
        let start = result.model.clone();
        let mut max_change: f64 = 0.0;
        for (index, sample) in samples.iter().enumerate() {
            let before = result.model.clone();
            // Steps 3–4: x_i = s_i, y_in = b + sum(x_i*w_i).
            let net = before.net(&sample.inputs);
            let error = sample.target - net;
            // Step 5: all updates use the SAME pre-update linear net input.
            let delta_bias = options.alpha * error;
            let delta_weights: Vec<f64> = sample.inputs.iter().map(|x| delta_bias * x).collect();
            let after = Model {
                weights: before
                    .weights
                    .iter()
                    .zip(&delta_weights)
                    .map(|(w, d)| w + d)
                    .collect(),
                bias: before.bias + delta_bias,
            };
            // A failed update is not committed or reported as a completed step.
            // The guard bounds the final fixed-model SSE, even for large inputs.
            if [net, error, delta_bias, after.bias]
                .iter()
                .chain(delta_weights.iter())
                .chain(after.weights.iter())
                .any(|v| !v.is_finite() || v.abs() > 1e100)
            {
                result.status = Status::NumericalLimit;
                result.failed_at = Some((epoch, index + 1));
                return Ok(result);
            }
            let change = delta_weights
                .iter()
                .map(|v| v.abs())
                .fold(delta_bias.abs(), f64::max);
            max_change = max_change.max(change);
            result.model = after.clone();
            result.steps.push(Step {
                epoch,
                pattern: index + 1,
                before,
                net,
                output: activate(net),
                error,
                delta_weights,
                delta_bias,
                after,
                max_change: change,
            });
        }
        let net_change = result
            .model
            .weights
            .iter()
            .zip(&start.weights)
            .map(|(a, b)| (a - b).abs())
            .fold((result.model.bias - start.bias).abs(), f64::max);
        let (sse, correct) = diagnostics(samples, &result.model);
        result.epochs.push(Epoch {
            number: epoch,
            max_change,
            net_change,
            sse,
            correct,
            model: result.model.clone(),
        });
        // Step 6: strict comparison; includes bias; check only after a full epoch.
        if max_change < options.tolerance {
            result.status = Status::ToleranceMet;
            break;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(epochs: usize) -> Options {
        Options {
            alpha: 0.2,
            tolerance: 0.001,
            max_epochs: epochs,
        }
    }
    fn zero(n: usize) -> Model {
        Model {
            weights: vec![0.0; n],
            bias: 0.0,
        }
    }
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
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn two_hand_worked_bipolar_patterns_use_linear_output() {
        let result = train(&and(), zero(2), options(1)).unwrap();
        let a = &result.steps[0];
        assert_eq!(a.output, 1); // Already correctly classified, but MUST still update.
        assert_eq!(a.after.weights, vec![0.2, 0.2]);
        close(a.after.bias, 0.2);
        let b = &result.steps[1];
        close(b.net, 0.2);
        close(b.error, -1.2);
        close(b.after.weights[0], -0.04);
        close(b.after.weights[1], 0.44);
        close(b.after.bias, -0.04);
    }

    #[test]
    fn tolerance_is_strict_includes_bias_and_waits_for_a_complete_epoch() {
        let samples = vec![Sample {
            inputs: vec![0.],
            target: 1.,
        }];
        let result = train(
            &samples,
            zero(1),
            Options {
                tolerance: 0.2,
                ..options(1)
            },
        )
        .unwrap();
        assert_eq!(result.status, Status::EpochLimit);
        close(result.epochs[0].max_change, 0.2);
        let samples = vec![
            Sample {
                inputs: vec![1.],
                target: 1.,
            },
            Sample {
                inputs: vec![-1.],
                target: -1.,
            },
        ];
        let result = train(
            &samples,
            zero(1),
            Options {
                tolerance: 1.,
                ..options(10)
            },
        )
        .unwrap();
        assert_eq!(result.status, Status::ToleranceMet);
        assert_eq!(result.steps.len(), 2);
    }

    #[test]
    fn stationary_epoch_and_correct_classification_do_not_replace_step_six() {
        let result = train(&and(), zero(2), options(100)).unwrap();
        let last = result.epochs.last().unwrap();
        assert_eq!(result.status, Status::EpochLimit);
        assert_eq!(last.correct, 4);
        assert!(last.net_change < 1e-12);
        assert!(last.max_change > 0.001);
    }

    #[test]
    fn custom_initialization_and_three_inputs() {
        let samples = vec![Sample {
            inputs: vec![1., -1., 1.],
            target: -1.,
        }];
        let result = train(
            &samples,
            Model {
                weights: vec![0.1, 0.2, 0.3],
                bias: -0.1,
            },
            options(1),
        )
        .unwrap();
        close(result.steps[0].net, 0.1);
        close(result.model.weights[1], 0.42);
        close(result.model.bias, -0.32);
        assert_eq!(activate(0.), 1);
        assert_eq!(activate(-0.00001), -1);
    }

    #[test]
    fn divergence_keeps_last_valid_model_and_marks_incomplete_epoch() {
        let result = train(
            &and(),
            zero(2),
            Options {
                alpha: 1.,
                ..options(1000)
            },
        )
        .unwrap();
        assert_eq!(result.status, Status::NumericalLimit);
        assert!(result.failed_at.is_some());
        assert!(diagnostics(&and(), &result.model).0.is_finite());
        assert!(result.steps.iter().all(|s| s.after.bias.is_finite()));
    }

    #[test]
    fn rejects_invalid_targets_shapes_numbers_and_excessive_trace() {
        let invalid = vec![Sample {
            inputs: vec![1.],
            target: 0.,
        }];
        assert!(train(&invalid, zero(1), options(1)).is_err());
        assert!(train(&and(), zero(3), options(1)).is_err());
        assert!(train(
            &and(),
            zero(2),
            Options {
                alpha: f64::NAN,
                ..options(1)
            }
        )
        .is_err());
        assert!(train(&and(), zero(2), options(100_000)).is_err());
    }
}
