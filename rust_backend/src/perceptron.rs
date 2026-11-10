//! Binary, bipolar sign, and threshold-band perceptron rules.

use super::{validate as validate_adaline, Model, Options as AdalineOptions, Sample};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Binary,
    Bipolar,
    BipolarError,
    ThresholdBand,
}

impl Mode {
    pub fn activate(self, net: f64, theta: f64) -> i8 {
        match self {
            Self::Binary => i8::from(net >= 0.0),
            Self::Bipolar | Self::BipolarError => super::activate(net),
            Self::ThresholdBand => {
                if net > theta {
                    1
                } else if net < -theta {
                    -1
                } else {
                    0
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub mode: Mode,
    pub alpha: f64,
    pub theta: f64,
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
    pub changed: bool,
}

#[derive(Debug)]
pub struct Epoch {
    pub number: usize,
    pub max_change: f64,
    pub net_change: f64,
    pub sse: f64,
    pub correct: usize,
    pub model: Model,
    pub mistakes: usize,
    pub updates: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Status {
    WeightsStable,
    EpochLimit,
    NumericalLimit,
}

impl Status {
    pub fn name(&self) -> &'static str {
        match self {
            Self::WeightsStable => "weights_stable",
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

pub fn validate(samples: &[Sample], model: &Model, options: Options) -> Result<(), String> {
    if !options.alpha.is_finite() || options.alpha <= 0.0 || options.alpha > 1.0 {
        return Err("Perceptron learning rate must satisfy 0 < rate <= 1.".into());
    }
    if !options.theta.is_finite() || options.theta < 0.0 || options.theta > 1e50 {
        return Err("Theta must be finite and between 0 and 1e50.".into());
    }
    if options.mode != Mode::ThresholdBand && options.theta != 0.0 {
        return Err("Binary and bipolar sign modes use threshold zero; set theta to 0.".into());
    }
    for sample in samples {
        let valid = match options.mode {
            Mode::Binary => sample.target == 0.0 || sample.target == 1.0,
            Mode::Bipolar | Mode::BipolarError | Mode::ThresholdBand => {
                sample.target == -1.0 || sample.target == 1.0
            }
        };
        if !valid {
            return Err(match options.mode {
                Mode::Binary => "Binary perceptron targets must be 0 or 1.".into(),
                Mode::Bipolar | Mode::BipolarError => {
                    "Bipolar perceptron targets must be -1 or +1.".into()
                }
                Mode::ThresholdBand => "Threshold-band targets must be -1 or +1.".into(),
            });
        }
    }
    // Reuse the common input/model/trace validation with canonical bipolar labels.
    let canonical: Vec<Sample> = samples
        .iter()
        .map(|s| Sample {
            inputs: s.inputs.clone(),
            target: if s.target == 0.0 { -1.0 } else { s.target },
        })
        .collect();
    validate_adaline(
        &canonical,
        model,
        AdalineOptions {
            alpha: options.alpha,
            tolerance: 1.0,
            max_epochs: options.max_epochs,
        },
    )
}

pub fn diagnostics(samples: &[Sample], model: &Model, options: Options) -> (f64, usize) {
    let mut sse = 0.0;
    let mut correct = 0;
    for sample in samples {
        let output = f64::from(
            options
                .mode
                .activate(model.net(&sample.inputs), options.theta),
        );
        sse += (sample.target - output).powi(2);
        correct += usize::from(output == sample.target);
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
    for epoch in 1..=options.max_epochs {
        let start = result.model.clone();
        let mut max_change: f64 = 0.0;
        let mut mistakes = 0;
        let mut updates = 0;
        for (index, sample) in samples.iter().enumerate() {
            let before = result.model.clone();
            let net = before.net(&sample.inputs);
            let output = options.mode.activate(net, options.theta);
            let error = sample.target - f64::from(output);
            let delta_bias = match options.mode {
                Mode::Binary | Mode::BipolarError => options.alpha * error,
                Mode::Bipolar | Mode::ThresholdBand => {
                    if error != 0.0 {
                        options.alpha * sample.target
                    } else {
                        0.0
                    }
                }
            };
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
            let changed = after.bias != before.bias
                || after
                    .weights
                    .iter()
                    .zip(&before.weights)
                    .any(|(a, b)| a != b);
            mistakes += usize::from(error != 0.0);
            updates += usize::from(changed);
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
                output,
                error,
                delta_weights,
                delta_bias,
                after,
                max_change: change,
                changed,
            });
        }
        let net_change = result
            .model
            .weights
            .iter()
            .zip(&start.weights)
            .map(|(a, b)| (a - b).abs())
            .fold((result.model.bias - start.bias).abs(), f64::max);
        let (sse, correct) = diagnostics(samples, &result.model, options);
        result.epochs.push(Epoch {
            number: epoch,
            max_change,
            net_change,
            sse,
            correct,
            model: result.model.clone(),
            mistakes,
            updates,
        });
        // Step 6: no coefficient changed ANYWHERE in the completed pass.
        // Epoch endpoint equality is insufficient: individual updates may cancel.
        if updates == 0 {
            result.status = Status::WeightsStable;
            break;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options(mode: Mode, max_epochs: usize) -> Options {
        Options {
            mode,
            alpha: 1.,
            theta: 0.,
            max_epochs,
        }
    }
    fn zero(n: usize) -> Model {
        Model {
            weights: vec![0.; n],
            bias: 0.,
        }
    }
    fn and() -> Vec<Sample> {
        vec![
            (vec![1., 1.], 1.),
            (vec![1., 0.], 0.),
            (vec![0., 1.], 0.),
            (vec![0., 0.], 0.),
        ]
        .into_iter()
        .map(|(inputs, target)| Sample { inputs, target })
        .collect()
    }

    #[test]
    fn hand_worked_binary_and_requires_a_complete_unchanged_epoch() {
        let r = train(&and(), zero(2), options(Mode::Binary, 100)).unwrap();
        assert_eq!(r.steps[0].output, 1);
        assert!(!r.steps[0].changed);
        assert_eq!(r.steps[1].delta_weights, vec![-1., 0.]);
        assert_eq!(r.steps[1].after.bias, -1.);
        let eighth = &r.steps[7];
        assert_eq!(eighth.after.weights, vec![-1., 0.]);
        assert_eq!(eighth.after.bias, -2.);
        assert_eq!(r.status, Status::WeightsStable);
        assert_eq!(r.epochs.len(), 8);
        assert_eq!(r.model.weights, vec![1., 2.]);
        assert_eq!(r.model.bias, -3.);
        assert_eq!(r.epochs[6].correct, 4);
        assert!(r.epochs[6].updates > 0);
        assert_eq!(r.epochs[7].updates, 0);
    }

    #[test]
    fn binary_zero_and_threshold_band_inclusive_boundaries_are_distinct() {
        assert_eq!(Mode::Binary.activate(0., 0.), 1);
        assert_eq!(Mode::Binary.activate(-0.125, 0.), 0);
        for net in [-0.25, 0., 0.25] {
            assert_eq!(Mode::ThresholdBand.activate(net, 0.25), 0);
        }
        assert_eq!(Mode::ThresholdBand.activate(0.5, 0.25), 1);
        assert_eq!(Mode::ThresholdBand.activate(-0.5, 0.25), -1);
    }

    #[test]
    fn bipolar_sign_has_no_neutral_output_and_uses_target_on_a_mistake() {
        for (net, output) in [(-0.125, -1), (-0.0, 1), (0.0, 1), (0.125, 1)] {
            assert_eq!(Mode::Bipolar.activate(net, 0.), output);
        }
        let samples = vec![Sample {
            inputs: vec![1.],
            target: -1.,
        }];
        let r = train(
            &samples,
            zero(1),
            Options {
                alpha: 0.25,
                ..options(Mode::Bipolar, 1)
            },
        )
        .unwrap();
        assert_eq!(r.steps[0].output, 1);
        assert_eq!(r.steps[0].error, -2.);
        assert_eq!(r.steps[0].delta_bias, -0.25);
        assert_eq!(r.model.weights, vec![-0.25]);
        assert_eq!(r.model.bias, -0.25);
        assert!(validate(
            &samples,
            &zero(1),
            Options {
                theta: 0.1,
                ..options(Mode::Bipolar, 1)
            }
        )
        .is_err());
    }

    #[test]
    fn hand_worked_bipolar_and_finishes_with_an_unchanged_pass() {
        let samples = vec![
            (vec![1., 1.], 1.),
            (vec![1., -1.], -1.),
            (vec![-1., 1.], -1.),
            (vec![-1., -1.], -1.),
        ]
        .into_iter()
        .map(|(inputs, target)| Sample { inputs, target })
        .collect::<Vec<_>>();
        let r = train(&samples, zero(2), options(Mode::Bipolar, 100)).unwrap();
        assert_eq!(r.status, Status::WeightsStable);
        assert_eq!(r.epochs.len(), 3);
        assert_eq!(r.model.weights, vec![1., 1.]);
        assert_eq!(r.model.bias, -1.);
        assert_eq!(r.epochs[0].model.weights, vec![0., 0.]);
        assert_eq!(r.epochs[0].model.bias, -2.);
        assert_eq!(r.epochs[1].correct, 4);
        assert_eq!(r.epochs[1].updates, 1);
        assert_eq!(r.epochs[2].updates, 0);
        assert!(r.steps.iter().all(|s| s.output == -1 || s.output == 1));
        assert!(train(&and(), zero(2), options(Mode::Bipolar, 1)).is_err());
    }

    #[test]
    fn band_uses_alpha_target_rather_than_alpha_classification_error() {
        let samples = vec![Sample {
            inputs: vec![1.],
            target: 1.,
        }];
        let r = train(
            &samples,
            Model {
                weights: vec![0.],
                bias: -0.5,
            },
            Options {
                alpha: 0.25,
                ..options(Mode::ThresholdBand, 1)
            },
        )
        .unwrap();
        assert_eq!(r.steps[0].output, -1);
        assert_eq!(r.steps[0].error, 2.);
        assert_eq!(r.steps[0].delta_bias, 0.25);
        assert_eq!(r.model.weights, vec![0.25]);
        assert_eq!(r.model.bias, -0.25);
    }

    #[test]
    fn canceling_updates_at_epoch_end_do_not_imply_stability() {
        for (mode, targets) in [
            (Mode::Binary, [0., 1.]),
            (Mode::Bipolar, [-1., 1.]),
            (Mode::ThresholdBand, [1., -1.]),
        ] {
            let samples: Vec<Sample> = targets
                .into_iter()
                .map(|target| Sample {
                    inputs: vec![0.],
                    target,
                })
                .collect();
            let r = train(&samples, zero(1), options(mode, 3)).unwrap();
            assert_eq!(r.status, Status::EpochLimit);
            assert!(r
                .epochs
                .iter()
                .all(|e| e.net_change == 0. && e.updates == 2 && e.mistakes == 2));
        }
    }

    #[test]
    fn invalid_modes_parameters_and_unsafe_attempted_steps_are_handled() {
        assert!(train(
            &and(),
            zero(2),
            Options {
                alpha: 1.1,
                ..options(Mode::Binary, 1)
            }
        )
        .is_err());
        assert!(train(
            &and(),
            zero(2),
            Options {
                theta: 0.1,
                ..options(Mode::Binary, 1)
            }
        )
        .is_err());
        assert!(train(&and(), zero(2), options(Mode::ThresholdBand, 1)).is_err());
        let samples = vec![Sample {
            inputs: vec![1e50, 1e50],
            target: 1.,
        }];
        let r = train(
            &samples,
            Model {
                weights: vec![1e50, 1e50],
                bias: 0.,
            },
            options(Mode::Binary, 1),
        )
        .unwrap();
        assert_eq!(r.status, Status::NumericalLimit);
        assert_eq!(r.failed_at, Some((1, 1)));
        assert!(r.steps.is_empty());
    }
}
