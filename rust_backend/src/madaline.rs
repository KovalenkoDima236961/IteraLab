use super::{
    activate, validate as validate_adaline, Model as Unit, Options, Sample, MAX_COORDINATES,
};

pub const MAX_HIDDEN_UNITS: usize = 32;

#[derive(Clone, Debug)]
pub struct Model {
    pub hidden: Vec<Unit>,
    pub output_weights: Vec<f64>,
    pub output_bias: f64,
}

#[derive(Debug)]
pub struct Response {
    pub nets: Vec<f64>,
    pub hidden_outputs: Vec<i8>,
    pub net: f64,
    pub output: i8,
}

impl Model {
    pub fn response(&self, inputs: &[f64]) -> Response {
        let nets: Vec<f64> = self.hidden.iter().map(|unit| unit.net(inputs)).collect();
        let hidden_outputs: Vec<i8> = nets.iter().map(|&net| activate(net)).collect();
        let net = hidden_outputs
            .iter()
            .zip(&self.output_weights)
            .fold(self.output_bias, |net, (&z, &v)| net + f64::from(z) * v);
        Response {
            nets,
            hidden_outputs,
            net,
            output: activate(net),
        }
    }
}

#[derive(Debug)]
pub struct Step {
    pub epoch: usize,
    pub pattern: usize,
    pub before: Model,
    pub response: Response,
    pub rule: &'static str,
    pub selected: Vec<bool>,
    pub errors: Vec<f64>,
    pub delta_weights: Vec<Vec<f64>>,
    pub delta_biases: Vec<f64>,
    pub after: Model,
    pub max_change: f64,
}

#[derive(Debug)]
pub struct Epoch {
    pub number: usize,
    pub max_change: f64,
    pub net_change: f64,
    /// Classification SSE: sum((target - thresholded_output)^2).
    pub sse: f64,
    pub correct: usize,
    pub model: Model,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Status {
    WeightsStable,
    ToleranceMet,
    EpochLimit,
    NumericalLimit,
}

impl Status {
    pub fn name(&self) -> &'static str {
        match self {
            Self::WeightsStable => "weights_stable",
            Self::ToleranceMet => "tolerance_met",
            Self::EpochLimit => "epoch_limit",
            Self::NumericalLimit => "numerical_limit",
        }
    }
}

pub struct Training {
    pub status: Status,
    pub failed_at: Option<(usize, usize)>,
    pub steps: Vec<Step>,
    pub epochs: Vec<Epoch>,
    pub model: Model,
}

pub fn validate(samples: &[Sample], model: &Model, options: Options) -> Result<(), String> {
    let units = model.hidden.len();
    if !(1..=MAX_HIDDEN_UNITS).contains(&units) {
        return Err(format!("Use 1 to {MAX_HIDDEN_UNITS} hidden ADALINE units."));
    }
    if model.output_weights.len() != units {
        return Err("Provide one fixed output weight per hidden unit.".into());
    }
    for unit in &model.hidden {
        validate_adaline(samples, unit, options)?;
    }
    if model
        .output_weights
        .iter()
        .chain([&model.output_bias])
        .any(|v| !v.is_finite() || v.abs() > 1e50)
    {
        return Err(
            "Fixed output weights and bias must be finite and at most 1e50 in magnitude.".into(),
        );
    }
    if samples.len() * options.max_epochs * model.hidden[0].weights.len() * units > MAX_COORDINATES
    {
        return Err(
            "Reduce the epoch limit: the full hidden-unit trace supports 1,000,000 input coordinates."
                .into(),
        );
    }
    Ok(())
}

pub fn diagnostics(samples: &[Sample], model: &Model) -> (f64, usize) {
    let mut sse = 0.0;
    let mut correct = 0;
    for sample in samples {
        let output = f64::from(model.response(&sample.inputs).output);
        sse += (sample.target - output).powi(2);
        correct += usize::from(output == sample.target);
    }
    (sse, correct)
}

pub fn train(samples: &[Sample], initial: Model, options: Options) -> Result<Training, String> {
    validate(samples, &initial, options)?;
    let units = initial.hidden.len();
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
        for (index, sample) in samples.iter().enumerate() {
            let before = result.model.clone();
            let response = before.response(&sample.inputs);
            let mut after = before.clone();
            let mut selected = vec![false; units];
            let rule = if sample.target == f64::from(response.output) {
                "correct"
            } else if sample.target == 1.0 {
                // Equal distances select the lowest-index unit. Select exactly one.
                let mut closest = 0;
                for j in 1..units {
                    if response.nets[j].abs() < response.nets[closest].abs() {
                        closest = j;
                    }
                }
                selected[closest] = true;
                "positive_closest"
            } else {
                // f(0)=+1: zero-net units are active and must also be corrected.
                // Excluding them falsely declares zero-initialized AND stable.
                selected = response.nets.iter().map(|&net| net >= 0.0).collect();
                "negative_positive"
            };
            let mut errors = vec![0.0; units];
            let mut delta_biases = vec![0.0; units];
            let mut delta_weights = vec![vec![0.0; sample.inputs.len()]; units];
            for j in 0..units {
                if selected[j] {
                    errors[j] = sample.target - response.nets[j];
                    delta_biases[j] = options.alpha * errors[j];
                    after.hidden[j].bias += delta_biases[j];
                    for (i, &x) in sample.inputs.iter().enumerate() {
                        delta_weights[j][i] = delta_biases[j] * x;
                        after.hidden[j].weights[i] += delta_weights[j][i];
                    }
                }
            }
            let bad = response
                .nets
                .iter()
                .chain([&response.net])
                .chain(errors.iter())
                .chain(delta_biases.iter())
                .chain(delta_weights.iter().flatten())
                .chain(
                    after
                        .hidden
                        .iter()
                        .flat_map(|u| u.weights.iter().chain([&u.bias])),
                )
                .any(|v| !v.is_finite() || v.abs() > 1e100);
            if bad {
                result.status = Status::NumericalLimit;
                result.failed_at = Some((epoch, index + 1));
                return Ok(result);
            }
            let change = delta_biases
                .iter()
                .chain(delta_weights.iter().flatten())
                .map(|v| v.abs())
                .fold(0.0, f64::max);
            max_change = max_change.max(change);
            result.model = after.clone();
            result.steps.push(Step {
                epoch,
                pattern: index + 1,
                before,
                response,
                rule,
                selected,
                errors,
                delta_weights,
                delta_biases,
                after,
                max_change: change,
            });
        }
        let net_change = result
            .model
            .hidden
            .iter()
            .zip(&start.hidden)
            .flat_map(|(a, b)| {
                a.weights
                    .iter()
                    .chain([&a.bias])
                    .zip(b.weights.iter().chain([&b.bias]))
            })
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let (sse, correct) = diagnostics(samples, &result.model);
        result.epochs.push(Epoch {
            number: epoch,
            max_change,
            net_change,
            sse,
            correct,
            model: result.model.clone(),
        });
        // Step 8: stopped changes or acceptable level, tested after the whole epoch.
        if max_change == 0.0 {
            result.status = Status::WeightsStable;
            break;
        }
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
    fn model(biases: [f64; 2]) -> Model {
        Model {
            hidden: biases
                .into_iter()
                .map(|bias| Unit {
                    weights: vec![0., 0.],
                    bias,
                })
                .collect(),
            output_weights: vec![0.5, 0.5],
            output_bias: 0.5,
        }
    }
    fn run(biases: [f64; 2], target: f64) -> Training {
        train(
            &[Sample {
                inputs: vec![1., -1.],
                target,
            }],
            model(biases),
            Options {
                alpha: 0.2,
                tolerance: 0.001,
                max_epochs: 1,
            },
        )
        .unwrap()
    }
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    fn many_units(biases: &[f64]) -> Model {
        Model {
            hidden: biases
                .iter()
                .map(|&bias| Unit {
                    weights: vec![0., 0.],
                    bias,
                })
                .collect(),
            output_weights: vec![0.5; biases.len()],
            output_bias: (biases.len() as f64 - 1.) * 0.5,
        }
    }

    #[test]
    fn positive_target_searches_all_units_and_breaks_ties_by_lowest_index() {
        let samples = vec![Sample {
            inputs: vec![1., -1.],
            target: 1.,
        }];
        let r = train(
            &samples,
            many_units(&[-0.8, -0.1, -0.1, -0.6]),
            Options {
                alpha: 0.2,
                tolerance: 0.001,
                max_epochs: 1,
            },
        )
        .unwrap();
        assert_eq!(r.steps[0].selected, vec![false, true, false, false]);
        close(r.model.hidden[1].bias, 0.12);
        close(r.model.hidden[1].weights[1], -0.22);
        let r = train(
            &samples,
            many_units(&[-0.8, -0.5, -0.1]),
            Options {
                alpha: 0.2,
                tolerance: 0.001,
                max_epochs: 1,
            },
        )
        .unwrap();
        assert_eq!(r.steps[0].selected, vec![false, false, true]);
    }

    #[test]
    fn negative_target_updates_positive_units_beyond_the_first_two() {
        let samples = vec![Sample {
            inputs: vec![1., -1.],
            target: -1.,
        }];
        let r = train(
            &samples,
            many_units(&[0.25, -0.5, 0.75, 0., 0.125]),
            Options {
                alpha: 0.2,
                tolerance: 0.001,
                max_epochs: 1,
            },
        )
        .unwrap();
        let s = &r.steps[0];
        assert_eq!(s.selected, vec![true, false, true, true, true]);
        close(s.delta_biases[4], -0.225);
        close(s.after.hidden[4].weights[0], -0.225);
        close(r.epochs[0].max_change, 0.35);
        assert_eq!(r.model.output_weights, vec![0.5; 5]);
        assert_eq!(r.model.output_bias, 2.);
    }

    #[test]
    fn validates_hidden_counts_output_shape_and_total_trace_size() {
        let samples = vec![Sample {
            inputs: vec![1., -1.],
            target: -1.,
        }];
        let options = Options {
            alpha: 0.2,
            tolerance: 0.001,
            max_epochs: 1,
        };
        assert!(validate(&samples, &many_units(&[]), options).is_err());
        assert!(validate(&samples, &many_units(&[0.; 33]), options).is_err());
        let mut model = many_units(&[0.; 3]);
        model.output_weights.pop();
        assert!(validate(&samples, &model, options).is_err());
        assert!(validate(
            &samples,
            &many_units(&[0.; 32]),
            Options {
                max_epochs: 20_000,
                ..options
            }
        )
        .is_err());
        assert!(validate(&samples, &many_units(&[-0.2]), options).is_ok());
        assert!(validate(&samples, &many_units(&[-0.2; 32]), options).is_ok());
    }

    #[test]
    fn correct_patterns_never_update() {
        let r = run([0.2, -0.3], 1.);
        assert_eq!(r.steps[0].rule, "correct");
        assert_eq!(r.steps[0].selected, [false, false]);
        assert_eq!(r.status, Status::WeightsStable);
        assert_eq!(r.model.hidden[0].weights, vec![0., 0.]);
    }

    #[test]
    fn positive_target_selects_closest_negative_net_and_ties_select_z1() {
        let r = run([-0.4, -0.1], 1.);
        let s = &r.steps[0];
        assert_eq!(s.selected, [false, true]);
        close(s.errors[1], 1.1);
        close(s.after.hidden[1].bias, 0.12);
        close(s.after.hidden[1].weights[0], 0.22);
        close(s.after.hidden[1].weights[1], -0.22);
        close(s.after.hidden[0].bias, -0.4);
        assert_eq!(run([-0.2, -0.2], 1.).steps[0].selected, [true, false]);
    }

    #[test]
    fn negative_target_updates_all_positive_units_from_the_same_old_response() {
        let r = run([0.2, 0.4], -1.);
        let s = &r.steps[0];
        assert_eq!(s.selected, [true, true]);
        close(s.delta_biases[0], -0.24);
        close(s.delta_biases[1], -0.28);
        close(s.after.hidden[0].bias, -0.04);
        close(s.after.hidden[1].bias, 0.12);
        assert_eq!(r.model.output_weights, [0.5, 0.5]);
        assert_eq!(r.model.output_bias, 0.5);
        assert_eq!(run([-0.2, 0.4], -1.).steps[0].selected, [false, true]);
    }

    #[test]
    fn zero_maps_to_positive_class_and_is_corrected_for_negative_targets() {
        let r = run([0., 0.], -1.);
        assert_eq!(r.steps[0].response.hidden_outputs, [1, 1]);
        assert_eq!(r.steps[0].selected, [true, true]);
        assert_eq!(r.status, Status::EpochLimit);
        assert_eq!(r.epochs[0].correct, 1);
    }

    #[test]
    fn acceptable_change_is_strict_includes_bias_and_waits_for_complete_epoch() {
        let samples = vec![Sample {
            inputs: vec![0., 0.],
            target: -1.,
        }];
        let r = train(
            &samples,
            model([0.25, -0.5]),
            Options {
                alpha: 0.2,
                tolerance: 0.25,
                max_epochs: 1,
            },
        )
        .unwrap();
        assert_eq!(r.status, Status::EpochLimit);
        close(r.epochs[0].max_change, 0.25);
        let r = train(
            &samples,
            model([0.25, -0.5]),
            Options {
                alpha: 0.2,
                tolerance: 0.26,
                max_epochs: 10,
            },
        )
        .unwrap();
        assert_eq!(r.status, Status::ToleranceMet);
        assert_eq!(r.epochs.len(), 1);
        let samples = vec![
            Sample {
                inputs: vec![1., -1.],
                target: 1.,
            },
            Sample {
                inputs: vec![1., -1.],
                target: -1.,
            },
        ];
        let r = train(
            &samples,
            model([0.2, -0.3]),
            Options {
                alpha: 0.2,
                tolerance: 0.001,
                max_epochs: 1,
            },
        )
        .unwrap();
        assert_eq!(r.steps.len(), 2); // The first correct pattern must not stop the epoch.
        assert!(r.epochs[0].max_change > 0.);
    }

    #[test]
    fn invalid_shapes_output_parameters_and_failed_update_are_handled() {
        let samples = vec![Sample {
            inputs: vec![1e50, 1e50],
            target: -1.,
        }];
        let mut initial = model([0., 0.]);
        initial.hidden[0].weights = vec![1e50, 1e50];
        let r = train(
            &samples,
            initial.clone(),
            Options {
                alpha: 1e50,
                tolerance: 0.001,
                max_epochs: 1,
            },
        )
        .unwrap();
        assert_eq!(r.status, Status::NumericalLimit);
        assert_eq!(r.failed_at, Some((1, 1)));
        assert!(r.steps.is_empty());
        assert_eq!(r.model.hidden[0].weights, initial.hidden[0].weights);
        initial.output_bias = f64::NAN;
        assert!(validate(
            &samples,
            &initial,
            Options {
                alpha: 0.2,
                tolerance: 0.001,
                max_epochs: 1
            }
        )
        .is_err());
    }
}
