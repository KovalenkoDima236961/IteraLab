use std::io::Write;

pub type Matrix = Vec<Vec<f64>>;

#[derive(Clone, Debug)]
pub struct Request {
    pub kind: String,
    pub s: Matrix,
    pub t: Matrix,
    pub w: Matrix,
    pub x: Matrix,
    pub bias: Matrix,
    pub expected: Matrix,
    pub bias_enabled: bool,
    pub tolerance: f64,
    pub activation: String,
}

#[derive(Debug)]
pub struct NamedMatrix {
    pub name: String,
    pub data: Matrix,
    pub columns: usize,
}
#[derive(Debug)]
pub struct Step {
    pub title: String,
    pub detail: String,
    pub matrices: Vec<String>,
}
#[derive(Default, Debug)]
pub struct Report {
    pub kind: String,
    pub matrices: Vec<NamedMatrix>,
    pub scalars: Vec<(String, f64)>,
    pub notes: Vec<(String, String)>,
    pub steps: Vec<Step>,
}

impl Report {
    fn matrix(&mut self, name: &str, data: Matrix) {
        let columns = data.first().map_or(0, Vec::len);
        self.matrices.push(NamedMatrix {
            name: name.into(),
            data,
            columns,
        });
    }
    fn scalar(&mut self, name: &str, value: f64) {
        self.scalars.push((name.into(), value));
    }
    fn note(&mut self, name: &str, value: impl Into<String>) {
        self.notes.push((name.into(), value.into()));
    }
    fn step(&mut self, title: impl Into<String>, detail: impl Into<String>, names: Vec<String>) {
        self.steps.push(Step {
            title: title.into(),
            detail: detail.into(),
            matrices: names,
        });
    }
}

fn zeros(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}
pub fn transpose(a: &Matrix) -> Matrix {
    (0..a[0].len())
        .map(|j| a.iter().map(|row| row[j]).collect())
        .collect()
}
pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    a.iter()
        .map(|row| {
            (0..b[0].len())
                .map(|j| row.iter().enumerate().map(|(k, x)| x * b[k][j]).sum())
                .collect()
        })
        .collect()
}
fn outer(a: &[f64], b: &[f64]) -> Matrix {
    a.iter()
        .map(|x| b.iter().map(|y| x * y).collect())
        .collect()
}
fn vector_text(v: &[f64]) -> String {
    format!(
        "({})",
        v.iter()
            .map(|x| format!("{x}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn validate_matrix(a: &Matrix, name: &str, max_rows: usize, limit: f64) -> Result<(), String> {
    let n = a.first().map_or(0, Vec::len);
    if a.is_empty() || a.len() > max_rows || n == 0 || n > 32 {
        return Err(format!("{name}: use 1-{max_rows} rows and 1-32 columns."));
    }
    if a.iter().any(|row| row.len() != n) {
        return Err(format!("{name}: every row must have {n} values."));
    }
    if a.iter()
        .flatten()
        .any(|v| !v.is_finite() || v.abs() > limit)
    {
        return Err(format!(
            "{name}: values must be finite and have magnitude at most {limit:e}."
        ));
    }
    Ok(())
}

impl Request {
    pub fn validate(&self) -> Result<(), String> {
        if !["binary", "sign", "bipolar"].contains(&self.activation.as_str()) {
            return Err("Unknown recall activation.".into());
        }
        match self.kind.as_str() {
            "matrix_product" => {
                validate_matrix(&self.s, "A", 128, 1e50)?;
                validate_matrix(&self.t, "B", 32, 1e50)?;
                if self.s[0].len() != self.t.len() {
                    return Err("A columns must equal B rows.".into());
                }
            }
            "mcp" => {
                validate_matrix(&self.s, "Inputs", 128, 1e50)?;
                validate_matrix(&self.w, "Neuron weights", 32, 1e50)?;
                validate_matrix(&self.t, "Thresholds", 1, 1e50)?;
                let (n, m) = (self.s[0].len(), self.w.len());
                if self.w[0].len() != n + m || self.t[0].len() != m {
                    return Err("For n inputs and m neurons, use m by n+m weights and one row of m thresholds.".into());
                }
                if self.s.iter().flatten().any(|&v| v != 0. && v != 1.) {
                    return Err("McCulloch-Pitts inputs must be 0 or 1.".into());
                }
                if self
                    .w
                    .iter()
                    .enumerate()
                    .any(|(j, row)| row[n + j..].iter().any(|&v| v != 0.))
                {
                    return Err("Weights on the same or later neuron must be zero.".into());
                }
                if !self.expected.is_empty() {
                    validate_matrix(&self.expected, "Expected outputs", 128, 1e50)?;
                    if self.expected.len() != self.s.len() || self.expected[0].len() != m {
                        return Err("Expected outputs must match pattern and neuron counts.".into());
                    }
                }
            }
            "correlation" | "orthogonality" => validate_matrix(&self.s, "X / S", 128, 1e50)?,
            "hebbian_memory" | "outer_product" => {
                validate_matrix(&self.s, "S", 128, 1e50)?;
                validate_matrix(&self.t, "T", 128, 1e50)?;
                if self.s.len() != self.t.len() {
                    return Err("S and T must contain the same number of training rows.".into());
                }
                if self.kind == "outer_product" && self.bias_enabled {
                    return Err("Outer-product construction has no bias; use Hebbian memory for separate bias updates.".into());
                }
            }
            "recall" => {
                validate_matrix(&self.w, "W", 32, 1e150)?;
                validate_matrix(&self.x, "X", 128, 1e50)?;
                if self.x[0].len() != self.w.len() {
                    return Err(
                        "Each recall input must have as many coordinates as W has rows.".into(),
                    );
                }
                if self.bias_enabled {
                    validate_matrix(&self.bias, "Bias", 1, 1e150)?;
                    if self.bias[0].len() != self.w[0].len() {
                        return Err("Bias must contain one value per output column of W.".into());
                    }
                }
                if !self.expected.is_empty() {
                    validate_matrix(&self.expected, "Expected targets", 128, 1e50)?;
                    if self.expected.len() != self.x.len()
                        || self.expected[0].len() != self.w[0].len()
                    {
                        return Err("Expected targets must have one row per recall input and one column per output.".into());
                    }
                }
            }
            _ => return Err("Unknown Lecture 08 calculation.".into()),
        }
        if !self.tolerance.is_finite() || self.tolerance < 0.0 || self.tolerance > 1e50 {
            return Err("Orthogonality tolerance must be finite and between 0 and 1e50.".into());
        }
        Ok(())
    }
}

struct Spectrum {
    values: Vec<f64>,
    vectors: Matrix,
    sweeps: Matrix,
    scale: f64,
    residuals: Matrix,
}

/// Cyclic Jacobi on a scaled symmetric matrix, with explicit convergence checking.
fn spectrum(matrix: &Matrix) -> Result<Spectrum, String> {
    let n = matrix.len();
    let scale = matrix.iter().flatten().map(|v| v.abs()).fold(0., f64::max);
    let divisor = if scale == 0. { 1. } else { scale };
    let original: Matrix = matrix
        .iter()
        .map(|row| row.iter().map(|v| v / divisor).collect())
        .collect();
    let mut a = original.clone();
    let mut v = zeros(n, n);
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.;
    }
    let offdiag = |a: &Matrix| {
        (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| a[i][j].abs()))
            .fold(0., f64::max)
    };
    let mut history = vec![vec![0., 0., offdiag(&a)]];
    let mut converged = offdiag(&a) <= 1e-14;
    for sweep in 1..=100 {
        if converged {
            break;
        }
        let mut rotations = 0;
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p][q];
                if apq.abs() <= 1e-14 {
                    continue;
                }
                let angle = 0.5 * (2. * apq).atan2(a[q][q] - a[p][p]);
                let (s, c) = angle.sin_cos();
                let (app, aqq) = (a[p][p], a[q][q]);
                for k in 0..n {
                    if k != p && k != q {
                        let (akp, akq) = (a[k][p], a[k][q]);
                        a[k][p] = c * akp - s * akq;
                        a[p][k] = a[k][p];
                        a[k][q] = s * akp + c * akq;
                        a[q][k] = a[k][q];
                    }
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
                a[p][p] = c * c * app - 2. * s * c * apq + s * s * aqq;
                a[q][q] = s * s * app + 2. * s * c * apq + c * c * aqq;
                a[p][q] = 0.;
                a[q][p] = 0.;
                rotations += 1;
            }
        }
        let remaining = offdiag(&a);
        history.push(vec![sweep as f64, rotations as f64, remaining]);
        converged = remaining <= 1e-14;
    }
    if !converged {
        return Err("Symmetric eigensolver did not converge; rescale the input matrix.".into());
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let eigen_norm: Vec<f64> = order
        .iter()
        .map(|&i| {
            if a[i][i] < 0. && a[i][i] > -1e-12 {
                0.
            } else {
                a[i][i]
            }
        })
        .collect();
    let values = eigen_norm.iter().map(|x| x * divisor).collect();
    let vectors: Matrix = v
        .iter()
        .map(|row| order.iter().map(|&j| row[j]).collect())
        .collect();
    let residuals = (0..n)
        .map(|j| {
            let residual = (0..n)
                .map(|i| {
                    ((0..n).map(|k| original[i][k] * vectors[k][j]).sum::<f64>()
                        - eigen_norm[j] * vectors[i][j])
                        .abs()
                })
                .fold(0., f64::max);
            let mut shifted = original.clone();
            for (i, row) in shifted.iter_mut().enumerate() {
                row[i] -= eigen_norm[j];
            }
            vec![eigen_norm[j] * divisor, residual, determinant(shifted)]
        })
        .collect();
    Ok(Spectrum {
        values,
        vectors,
        sweeps: history,
        scale: divisor,
        residuals,
    })
}

/// Partial-pivoted determinant; used on the normalized R-lambda*I certificate.
fn determinant(mut a: Matrix) -> f64 {
    let n = a.len();
    let mut det = 1.;
    for j in 0..n {
        let pivot = (j..n)
            .max_by(|&i, &k| a[i][j].abs().total_cmp(&a[k][j].abs()))
            .unwrap();
        if a[pivot][j] == 0. {
            return 0.;
        }
        if pivot != j {
            a.swap(pivot, j);
            det = -det;
        }
        let diagonal = a[j][j];
        det *= diagonal;
        for i in j + 1..n {
            let factor = a[i][j] / diagonal;
            for k in j + 1..n {
                a[i][k] -= factor * a[j][k];
            }
        }
    }
    det
}

fn correlation(req: &Request, r: &mut Report) -> Result<(), String> {
    let (p, n) = (req.s.len(), req.s[0].len());
    r.matrix("X", req.s.clone());
    let mut sum = zeros(n, n);
    for (k, row) in req.s.iter().enumerate() {
        let contribution = outer(row, row);
        for i in 0..n {
            for j in 0..n {
                sum[i][j] += contribution[i][j];
            }
        }
        let name = format!("outer_{}", k + 1);
        r.matrix(&name, contribution);
        r.step(
            format!("Pattern {}: x^T x", k + 1),
            format!(
                "x = {}; outer[i,j] = x[i] * x[j]. Dimensions: {n}x1 times 1x{n} = {n}x{n}.",
                vector_text(row)
            ),
            vec![name],
        );
    }
    let correlation: Matrix = sum
        .iter()
        .map(|row| row.iter().map(|v| v / p as f64).collect())
        .collect();
    r.matrix("sum_outer", sum);
    r.matrix("R", correlation.clone());
    r.step("Correlation matrix",format!("R = X^T X / P = sum(outer products) / {p}. No mean centering or automatic bias column is used."),vec!["sum_outer".into(),"R".into()]);
    let eig = spectrum(&correlation)?;
    let lambda_max = eig.values[0];
    let radius = eig.values.iter().map(|v| v.abs()).fold(0., f64::max);
    r.scalar("lambda_max", lambda_max);
    r.scalar("spectral_radius", radius);
    r.scalar("lecture_alpha_upper", radius * 0.5);
    if lambda_max > 0. {
        r.scalar("mean_lms_alpha_upper", 2. / lambda_max);
        r.scalar("mse_gradient_alpha_upper", 1. / lambda_max);
    }
    r.note("stability_rule", "For the mean LMS update w += alpha * mean(x*(t-xw)), error evolves by I-alpha*R, so 0<alpha<2/lambda_max on positive-eigenvalue directions. For gradient descent on mean squared error without the 1/2 factor, use 0<alpha<1/lambda_max. These are batch/mean rules, not guarantees for constant-rate sequential training. Include a constant input column when analyzing a trainable bias. If R=0, there is no data curvature constraint; nullspace weights are not identified. The handwritten alpha<lambda_max/2 rule is not a general stability bound: it increases when input magnitudes increase, whereas the correct upper bound decreases.");
    r.scalar("eigen_normalizer", eig.scale);
    r.scalar(
        "eigen_residual_max",
        eig.residuals.iter().map(|row| row[1]).fold(0., f64::max),
    );
    r.matrix("eigenvalues", vec![eig.values]);
    r.matrix("eigenvectors", eig.vectors);
    r.matrix("eigen_checks", eig.residuals);
    r.matrix("jacobi_sweeps", eig.sweeps);
    let equation = if n == 1 {
        format!("det(R-lambda I) = ({} - lambda) = 0.", correlation[0][0])
    } else if n == 2 {
        format!("det(R-lambda I) = ({} - lambda)({} - lambda) - ({})({}) = 0. Equivalently lambda^2 - ({})lambda + ({}) = 0.",correlation[0][0],correlation[1][1],correlation[0][1],correlation[1][0],correlation[0][0]+correlation[1][1],correlation[0][0]*correlation[1][1]-correlation[0][1]*correlation[1][0])
    } else {
        format!("Solve det(R-lambda I)=0 for this {n}x{n} symmetric R. Eigenvalues are computed by converged Jacobi diagonalization, whose diagonal entries are these roots.")
    };
    r.step(
        "Eigenvalues: det(R - lambda I) = 0",
        equation,
        vec![
            "R".into(),
            "eigenvalues".into(),
            "eigenvectors".into(),
            "eigen_checks".into(),
            "jacobi_sweeps".into(),
        ],
    );
    r.note("eigen_checks","Columns: lambda, max |(R/scale)q-(lambda/scale)q|, det((R-lambda I)/scale). Eigenvectors are columns, in eigenvalue order. Scaling prevents overflow in determinant checks.");
    r.note(
        "jacobi_sweeps",
        "Columns: sweep, rotations in that sweep, largest remaining normalized off-diagonal entry.",
    );
    r.note("learning_rule","Lecture 08 page 2: 0 < alpha < (1/2) max_i |lambda_i|. This is the lecture's literal rule, not an automatically substituted standard convergence bound.");
    r.step("Lecture-specific learning rate",if radius==0. {"The lecture's strict upper bound is 0. No positive alpha satisfies 0 < alpha < 0 for this zero correlation matrix.".into()}
        else {format!("Lecture 08 rule: 0 < alpha < 0.5 * max|lambda| = 0.5 * {radius} = {}. The inequality is strict.",radius*0.5)},vec!["eigenvalues".into()]);
    Ok(())
}

fn orthogonality(s: &Matrix, tolerance: f64, r: &mut Report) -> Matrix {
    let gram = multiply(s, &transpose(s));
    let mut pairs = Vec::new();
    let mut all = true;
    for i in 0..s.len() {
        for j in i + 1..s.len() {
            let yes = gram[i][j].abs() <= tolerance;
            all &= yes;
            pairs.push(vec![
                (i + 1) as f64,
                (j + 1) as f64,
                gram[i][j],
                f64::from(yes),
            ]);
        }
    }
    r.matrix("gram", gram.clone());
    r.matrices.push(NamedMatrix {
        name: "pairwise_dots".into(),
        data: pairs,
        columns: 4,
    });
    r.matrix(
        "squared_norms",
        vec![(0..s.len()).map(|i| gram[i][i]).collect()],
    );
    r.scalar("all_orthogonal", f64::from(all));
    r.scalar("orthogonality_tolerance", tolerance);
    r.scalar(
        "zero_vectors",
        (0..s.len()).filter(|&i| gram[i][i] == 0.).count() as f64,
    );
    r.note("pairwise_dots","Columns: pattern k, pattern p, dot product, orthogonal (1=yes, 0=no). Every distinct pair is shown once. Absolute tolerance is explicit; default 0 tests exact zero.");
    r.note("cross_talk","For bias-free Hebbian W: s_k W = ||s_k||^2 t_k + sum_{p != k}(s_k dot s_p)t_p. Nonzero pairwise dots introduce cross-talk terms; they need not cause a wrong thresholded output. Zero vectors are orthogonal to everything but cannot encode distinct nonzero targets without bias.");
    gram
}

fn recall(
    x: &Matrix,
    w: &Matrix,
    bias: &[f64],
    expected: &Matrix,
    prefix: &str,
    record_steps: bool,
    activation: &str,
    r: &mut Report,
) {
    let linear = multiply(x, w);
    let net: Matrix = linear
        .iter()
        .map(|row| row.iter().zip(bias).map(|(v, b)| v + b).collect())
        .collect();
    let output: Matrix = net
        .iter()
        .map(|row| {
            row.iter()
                .map(|&v| match activation {
                    "sign" => {
                        if v > 0. {
                            1.
                        } else if v < 0. {
                            -1.
                        } else {
                            0.
                        }
                    }
                    "bipolar" => {
                        if v >= 0. {
                            1.
                        } else {
                            -1.
                        }
                    }
                    _ => {
                        if v > 0. {
                            1.
                        } else {
                            0.
                        }
                    }
                })
                .collect()
        })
        .collect();
    let linear_key = format!("{prefix}linear");
    let net_key = format!("{prefix}net");
    let output_key = format!("{prefix}output");
    r.matrix(&linear_key, linear.clone());
    r.matrix(&net_key, net.clone());
    r.matrix(&output_key, output.clone());
    if !expected.is_empty() {
        let matches: Matrix = output
            .iter()
            .zip(expected)
            .map(|(a, b)| a.iter().zip(b).map(|(a, b)| f64::from(a == b)).collect())
            .collect();
        r.scalar(
            &format!("{prefix}correct_patterns"),
            matches
                .iter()
                .filter(|row| row.iter().all(|&v| v == 1.))
                .count() as f64,
        );
        r.scalar(
            &format!("{prefix}matched_elements"),
            matches.iter().flatten().filter(|&&v| v == 1.).count() as f64,
        );
        r.matrix(&format!("{prefix}expected"), expected.clone());
        r.matrix(&format!("{prefix}matches"), matches);
    }
    if record_steps {
        for (k, row) in x.iter().enumerate() {
            let terms: Matrix = w
                .iter()
                .zip(row)
                .map(|(weights, x)| weights.iter().map(|v| v * x).collect())
                .collect();
            let name = format!("terms_{}", k + 1);
            r.matrix(&name, terms);
            let query = format!("query_{}", k + 1);
            r.matrix(&query, vec![row.clone()]);
            let weighted = format!("weighted_{}", k + 1);
            r.matrix(&weighted, vec![net[k].clone()]);
            let prediction = format!("prediction_{}", k + 1);
            r.matrix(&prediction, vec![output[k].clone()]);
            r.step(format!("Recall input {}",k+1),format!("Each term[i,j] = x[i] * W[i,j]. Sum each output column, then add the separate bias vector if enabled. Weighted sums = {}. Apply {activation} activation; output = {}.",vector_text(&net[k]),vector_text(&output[k])),vec![query,"W".into(),name,"bias".into(),weighted,prediction]);
        }
    }
}

fn memory(req: &Request, r: &mut Report) {
    let (p, n, m) = (req.s.len(), req.s[0].len(), req.t[0].len());
    r.matrix("S", req.s.clone());
    r.matrix("T", req.t.clone());
    let mut weights = zeros(n, m);
    let mut bias = vec![0.; m];
    r.matrix("initial_W", weights.clone());
    r.matrix("initial_bias", vec![bias.clone()]);
    r.step("Zero initialization",format!("Initialize W to a zero {n}x{m} input-by-output matrix and maintain any enabled bias separately. Each training pair is processed exactly once."),vec!["initial_W".into(),"initial_bias".into()]);
    for k in 0..p {
        let delta = outer(&req.s[k], &req.t[k]);
        for i in 0..n {
            for j in 0..m {
                weights[i][j] += delta[i][j];
            }
        }
        if req.bias_enabled {
            for (b, t) in bias.iter_mut().zip(&req.t[k]) {
                *b += t;
            }
        }
        let contribution = format!("outer_{}", k + 1);
        let after = format!("weights_after_{}", k + 1);
        r.matrix(&contribution, delta);
        r.matrix(&after, weights.clone());
        let mut names = vec![contribution, after];
        if req.bias_enabled {
            let key = format!("bias_after_{}", k + 1);
            r.matrix(&key, vec![bias.clone()]);
            names.push(key);
        }
        r.step(
            format!("Training pair {}", k + 1),
            format!(
                "s = {}; t = {}. Delta W[i,j] = s[i] * t[j]; W_new = W_old + Delta W. {}",
                vector_text(&req.s[k]),
                vector_text(&req.t[k]),
                if req.bias_enabled {
                    "Separate bias: b_new[j] = b_old[j] + t[j]."
                } else {
                    "Bias is disabled. No learning-rate factor or repeated epochs are used."
                }
            ),
            names,
        );
    }
    let direct = multiply(&transpose(&req.s), &req.t);
    let difference = weights
        .iter()
        .flatten()
        .zip(direct.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    r.scalar("outer_product_difference", difference);
    r.scalar("outer_product_agreement", f64::from(difference == 0.));
    r.scalar("bias_enabled", f64::from(req.bias_enabled));
    r.matrix("W", weights.clone());
    r.matrix("bias", vec![bias.clone()]);
    r.matrix("W_from_matrix_product", direct);
    r.step("Final W and independent matrix-form check","W = sum_p s_p^T t_p = S^T T. The direct matrix product is compared with the online Hebbian/outer-product accumulation. Bias is maintained separately and never added as a hidden row of W.",vec!["W".into(),"W_from_matrix_product".into(),"bias".into()]);
    let gram = orthogonality(&req.s, req.tolerance, r);
    let cross: Matrix = gram
        .iter()
        .enumerate()
        .map(|(k, row)| {
            (0..m)
                .map(|j| {
                    (0..p)
                        .filter(|&q| q != k)
                        .map(|q| row[q] * req.t[q][j])
                        .sum()
                })
                .collect()
        })
        .collect();
    r.matrix("cross_talk", cross);
    recall(
        &req.s,
        &weights,
        &bias,
        &req.t,
        "training_",
        false,
        &req.activation,
        r,
    );
    r.step("Training-pattern recall and cross-talk",format!("Use the same final W on every training row. Cross-talk is sum_{{p != k}}(s_k dot s_p)t_p before bias. Recall uses the selected {} activation; inspect actual predictions.",req.activation),vec!["gram".into(),"pairwise_dots".into(),"cross_talk".into(),"training_net".into(),"training_output".into(),"training_expected".into()]);
}

fn mcp(req: &Request, r: &mut Report) {
    let (n, m) = (req.s[0].len(), req.w.len());
    r.matrix("X", req.s.clone());
    r.matrix("W", req.w.clone());
    r.matrix("thresholds", req.t.clone());
    let mut nets = Vec::new();
    let mut outputs = Vec::new();
    for (k, row) in req.s.iter().enumerate() {
        let mut sources = row.clone();
        sources.resize(n + m, 0.);
        let mut net_row = Vec::new();
        for j in 0..m {
            let terms: Vec<f64> = sources.iter().zip(&req.w[j]).map(|(x, w)| x * w).collect();
            let z: f64 = terms.iter().sum();
            let y = f64::from(z >= req.t[0][j]);
            sources[n + j] = y;
            net_row.push(z);
            let key = format!("terms_{}_{}", k + 1, j + 1);
            r.matrix(&key, vec![terms]);
            r.step(format!("Pattern {}, neuron {}",k+1,j+1),format!("z = sum of weighted inputs = {z}; threshold = {}. Output = 1 if z >= threshold, otherwise 0: {y}. Use this output only in later neurons. Weights and thresholds are fixed; no training updates occur.",req.t[0][j]),vec![key]);
        }
        nets.push(net_row);
        outputs.push(sources[n..].to_vec());
    }
    if !req.expected.is_empty() {
        let matches: Matrix = outputs
            .iter()
            .zip(&req.expected)
            .map(|(a, b)| a.iter().zip(b).map(|(a, b)| f64::from(a == b)).collect())
            .collect();
        r.scalar(
            "correct_patterns",
            matches
                .iter()
                .filter(|row| row.iter().all(|&v| v == 1.))
                .count() as f64,
        );
        r.matrix("expected", req.expected.clone());
        r.matrix("matches", matches);
    }
    r.matrix("net", nets);
    r.matrix("output", outputs);
}

pub fn calculate(req: &Request) -> Result<Report, String> {
    req.validate()?;
    let mut r = Report {
        kind: req.kind.clone(),
        ..Report::default()
    };
    match req.kind.as_str() {
        "correlation" => correlation(req, &mut r)?,
        "mcp" => mcp(req, &mut r),
        "matrix_product" => {
            r.matrix("A", req.s.clone());
            r.matrix("B", req.t.clone());
            r.matrix("A_transpose", transpose(&req.s));
            r.matrix("B_transpose", transpose(&req.t));
            let product = multiply(&req.s, &req.t);
            for (i, row) in req.s.iter().enumerate() {
                let terms: Matrix = (0..req.t[0].len())
                    .map(|j| {
                        row.iter()
                            .enumerate()
                            .map(|(k, v)| v * req.t[k][j])
                            .collect()
                    })
                    .collect();
                let key = format!("row_{}_terms", i + 1);
                r.matrix(&key, terms);
                r.step(format!("Product row {}",i+1),"Each displayed row lists the terms of one output-column dot product. Sum across it to get C[i,j] = sum_k A[i,k]*B[k,j].",vec![key]);
            }
            r.matrix("output", product);
        }
        "hebbian_memory" | "outer_product" => memory(req, &mut r),
        "orthogonality" => {
            r.matrix("S", req.s.clone());
            orthogonality(&req.s, req.tolerance, &mut r);
            r.step("All distinct pairwise dot products","Compute G = S S^T. Patterns k and p are orthogonal when |G[k,p]| <= the selected absolute tolerance. Diagonal entries are squared norms, not distinct-pair tests.",vec!["S".into(),"gram".into(),"pairwise_dots".into(),"squared_norms".into()]);
        }
        "recall" => {
            let bias = if req.bias_enabled {
                req.bias[0].clone()
            } else {
                vec![0.; req.w[0].len()]
            };
            r.matrix("X", req.x.clone());
            r.matrix("W", req.w.clone());
            r.matrix("bias", vec![bias.clone()]);
            r.scalar("bias_enabled", f64::from(req.bias_enabled));
            recall(
                &req.x,
                &req.w,
                &bias,
                &req.expected,
                "",
                true,
                &req.activation,
                &mut r,
            );
            r.step("Recall result",format!("Weighted sums = XW plus separate bias if enabled. Apply {} activation element by element. Binary: 1 for z>0, otherwise 0; sign: -1/0/+1; bipolar: +1 for z>=0, otherwise -1. Expected rows are compared literally.",req.activation),vec!["linear".into(),"net".into(),"output".into()]);
        }
        _ => unreachable!(),
    }
    let finite = r
        .matrices
        .iter()
        .flat_map(|m| m.data.iter().flatten())
        .chain(r.scalars.iter().map(|(_, v)| v))
        .all(|v| v.is_finite());
    if !finite {
        return Err(
            "The matrix calculation exceeded finite arithmetic. Rescale the inputs.".into(),
        );
    }
    Ok(r)
}

pub fn parse_request(mut lines: std::str::Lines<'_>) -> Result<Request, String> {
    let kind = lines
        .next()
        .ok_or("Missing Lecture 08 operation.")?
        .to_string();
    let flags: Vec<_> = lines
        .next()
        .ok_or("Missing Lecture 08 options.")?
        .split('\t')
        .collect();
    if flags.len() != 2 && flags.len() != 3 {
        return Err(
            "Expected bias flag, orthogonality tolerance and optional recall activation.".into(),
        );
    }
    let bias_enabled = match flags[0] {
        "0" => false,
        "1" => true,
        _ => return Err("Invalid bias flag.".into()),
    };
    let tolerance = flags[1]
        .parse()
        .map_err(|_| "Invalid orthogonality tolerance.")?;
    let mut matrices = Vec::new();
    for name in ["S", "T", "W", "X", "bias", "expected"] {
        let shape: Vec<_> = lines
            .next()
            .ok_or(format!("Missing {name} dimensions."))?
            .split('\t')
            .collect();
        if shape.len() != 2 {
            return Err(format!("Invalid {name} dimensions."));
        }
        let rows: usize = shape[0].parse().map_err(|_| "Invalid row count.")?;
        let cols: usize = shape[1].parse().map_err(|_| "Invalid column count.")?;
        if rows > 128 || cols > 32 || (rows == 0) != (cols == 0) {
            return Err(format!("{name}: invalid matrix shape."));
        }
        let values = lines.next().ok_or(format!("Missing {name} values."))?;
        let flat: Vec<f64> = if values.is_empty() {
            Vec::new()
        } else {
            values
                .split(',')
                .map(|s| s.parse().map_err(|_| format!("{name}: invalid number.")))
                .collect::<Result<_, _>>()?
        };
        if flat.len() != rows * cols {
            return Err(format!("{name}: value count does not match its shape."));
        }
        matrices.push(if cols == 0 {
            Vec::new()
        } else {
            flat.chunks(cols).map(|row| row.to_vec()).collect()
        });
    }
    if lines.any(|line| !line.is_empty()) {
        return Err("Unexpected extra Lecture 08 data.".into());
    }
    let mut it = matrices.into_iter();
    Ok(Request {
        kind,
        s: it.next().unwrap(),
        t: it.next().unwrap(),
        w: it.next().unwrap(),
        x: it.next().unwrap(),
        bias: it.next().unwrap(),
        expected: it.next().unwrap(),
        bias_enabled,
        tolerance,
        activation: flags.get(2).unwrap_or(&"binary").to_string(),
    })
}

fn escaped(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

pub fn write_report(mut out: impl Write, r: &Report) -> Result<(), std::io::Error> {
    writeln!(out, "L08\t{}", r.kind)?;
    for m in &r.matrices {
        let flat = m
            .data
            .iter()
            .flatten()
            .map(|v| format!("{v:.17e}"))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(
            out,
            "M\t{}\t{}\t{}\t{}",
            m.name,
            m.data.len(),
            m.columns,
            flat
        )?;
    }
    for (key, value) in &r.scalars {
        writeln!(out, "D\t{key}\t{value:.17e}")?;
    }
    for (key, value) in &r.notes {
        writeln!(out, "N\t{key}\t{}", escaped(value))?;
    }
    for s in &r.steps {
        writeln!(
            out,
            "S\t{}\t{}\t{}",
            escaped(&s.title),
            escaped(&s.detail),
            s.matrices.join(",")
        )?;
    }
    writeln!(out, "END")?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(kind: &str) -> Request {
        Request {
            kind: kind.into(),
            s: vec![],
            t: vec![],
            w: vec![],
            x: vec![],
            bias: vec![],
            expected: vec![],
            bias_enabled: false,
            tolerance: 0.,
            activation: "binary".into(),
        }
    }
    fn matrix<'a>(r: &'a Report, name: &str) -> &'a Matrix {
        &r.matrices.iter().find(|m| m.name == name).unwrap().data
    }
    fn scalar(r: &Report, name: &str) -> f64 {
        r.scalars.iter().find(|(key, _)| key == name).unwrap().1
    }
    fn lecture_memory(kind: &str) -> Request {
        Request {
            s: vec![
                vec![1., 0., 0., 0.],
                vec![1., 1., 0., 0.],
                vec![0., 0., 0., 1.],
                vec![0., 0., 1., 1.],
            ],
            t: vec![vec![1., 0.], vec![1., 0.], vec![0., 1.], vec![0., 1.]],
            ..request(kind)
        }
    }
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-11 * b.abs().max(1.), "{a} != {b}");
    }

    #[test]
    fn lecture_correlation_eigenvalues_and_literal_half_maximum_rule() {
        let r = calculate(&Request {
            s: vec![vec![1., 1.], vec![1., 0.], vec![0., 1.], vec![0., 0.]],
            ..request("correlation")
        })
        .unwrap();
        assert_eq!(matrix(&r, "R"), &vec![vec![0.5, 0.25], vec![0.25, 0.5]]);
        close(matrix(&r, "eigenvalues")[0][0], 0.75);
        close(matrix(&r, "eigenvalues")[0][1], 0.25);
        close(scalar(&r, "lecture_alpha_upper"), 0.375);
        assert!(scalar(&r, "eigen_residual_max") < 1e-12);
        assert!(matrix(&r, "eigen_checks")
            .iter()
            .all(|row| row[2].abs() < 1e-12));
    }

    #[test]
    fn spectrum_handles_zero_one_dimension_repeated_and_rotated_eigenvalues() {
        for (s, expected) in [
            (vec![vec![0., 0.]], vec![0., 0.]),
            (vec![vec![2.]], vec![4.]),
            (vec![vec![1., 1.], vec![1., 1.]], vec![2., 0.]),
            (
                vec![vec![1., 1., 0.], vec![0., 1., 1.], vec![1., 0., 1.]],
                vec![4. / 3., 1. / 3., 1. / 3.],
            ),
        ] {
            let r = calculate(&Request {
                s,
                ..request("correlation")
            })
            .unwrap();
            for (&actual, expected) in matrix(&r, "eigenvalues")[0].iter().zip(expected) {
                close(actual, expected);
            }
            assert!(scalar(&r, "eigen_residual_max") < 1e-12);
            let vectors = matrix(&r, "eigenvectors");
            let gram = multiply(&transpose(vectors), vectors);
            for (i, row) in gram.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    close(v, f64::from(i == j));
                }
            }
        }
    }

    #[test]
    fn scaled_and_32_coordinate_correlation_remain_finite() {
        let large = calculate(&Request {
            s: vec![
                vec![1e40, 1e40],
                vec![1e40, 0.],
                vec![0., 1e40],
                vec![0., 0.],
            ],
            ..request("correlation")
        })
        .unwrap();
        close(scalar(&large, "lecture_alpha_upper"), 0.375e80);
        let mut identity = zeros(32, 32);
        for (i, row) in identity.iter_mut().enumerate() {
            row[i] = 1.;
        }
        let r = calculate(&Request {
            s: identity,
            ..request("correlation")
        })
        .unwrap();
        assert_eq!(matrix(&r, "eigenvalues")[0], vec![1. / 32.; 32]);
    }

    #[test]
    fn lecture_hebbian_and_outer_product_results_match_and_recall_correctly() {
        let h = calculate(&lecture_memory("hebbian_memory")).unwrap();
        let o = calculate(&lecture_memory("outer_product")).unwrap();
        let expected = vec![vec![2., 0.], vec![1., 0.], vec![0., 1.], vec![0., 2.]];
        assert_eq!(matrix(&h, "W"), &expected);
        assert_eq!(matrix(&h, "W"), matrix(&o, "W"));
        assert_eq!(matrix(&h, "W"), matrix(&h, "W_from_matrix_product"));
        assert_eq!(
            matrix(&h, "weights_after_1"),
            &vec![vec![1., 0.], vec![0., 0.], vec![0., 0.], vec![0., 0.]]
        );
        assert_eq!(
            matrix(&h, "training_net"),
            &vec![vec![2., 0.], vec![3., 0.], vec![0., 2.], vec![0., 3.]]
        );
        assert_eq!(scalar(&h, "training_correct_patterns"), 4.);
        assert_eq!(scalar(&h, "all_orthogonal"), 0.);
        assert_eq!(
            matrix(&h, "cross_talk"),
            &vec![vec![1., 0.], vec![1., 0.], vec![0., 1.], vec![0., 1.]]
        );
    }

    #[test]
    fn arbitrary_real_inputs_multiple_outputs_and_separate_bias_updates() {
        let req = Request {
            s: vec![vec![2., -1.], vec![0.5, 3.]],
            t: vec![vec![1., 2., -1.], vec![-2., 0.5, 4.]],
            bias_enabled: true,
            ..request("hebbian_memory")
        };
        let r = calculate(&req).unwrap();
        assert_eq!(
            matrix(&r, "W"),
            &vec![vec![1., 4.25, 0.], vec![-7., -0.5, 13.]]
        );
        assert_eq!(matrix(&r, "bias"), &vec![vec![-1., 2.5, 3.]]);
        assert_eq!(matrix(&r, "bias_after_1"), &vec![vec![1., 2., -1.]]);
        let mut no_bias = req;
        no_bias.bias_enabled = false;
        let without = calculate(&no_bias).unwrap();
        assert_eq!(matrix(&r, "W"), matrix(&without, "W"));
        assert_eq!(matrix(&without, "bias"), &vec![vec![0., 0., 0.]]);
    }

    #[test]
    fn orthogonality_checks_every_distinct_pair_and_zero_vectors() {
        let req = Request {
            s: vec![vec![1., 0.], vec![0., 1.], vec![0., 0.]],
            ..request("orthogonality")
        };
        let r = calculate(&req).unwrap();
        assert_eq!(scalar(&r, "all_orthogonal"), 1.);
        assert_eq!(scalar(&r, "zero_vectors"), 1.);
        assert_eq!(matrix(&r, "pairwise_dots").len(), 3);
        let mut near = Request {
            s: vec![vec![1., 0.], vec![1e-8, 1.]],
            ..request("orthogonality")
        };
        assert_eq!(scalar(&calculate(&near).unwrap(), "all_orthogonal"), 0.);
        near.tolerance = 1e-7;
        assert_eq!(scalar(&calculate(&near).unwrap(), "all_orthogonal"), 1.);
        let singleton = calculate(&Request {
            s: vec![vec![0.]],
            ..request("orthogonality")
        })
        .unwrap();
        assert!(matrix(&singleton, "pairwise_dots").is_empty());
    }

    #[test]
    fn recall_strict_zero_negative_values_optional_bias_and_expected_comparison() {
        let req = Request {
            w: vec![vec![1., -1., 0.]],
            x: vec![vec![1.], vec![0.], vec![-1.]],
            expected: vec![vec![1., 0., 0.], vec![0., 0., 0.], vec![0., 1., 0.]],
            ..request("recall")
        };
        let r = calculate(&req).unwrap();
        assert_eq!(matrix(&r, "output"), &req.expected);
        assert_eq!(scalar(&r, "correct_patterns"), 3.);
        let mut biased = req;
        biased.bias_enabled = true;
        biased.bias = vec![vec![0., 0., 0.5]];
        let r = calculate(&biased).unwrap();
        assert!(matrix(&r, "output").iter().all(|row| row[2] == 1.));
        assert_eq!(matrix(&r, "linear")[0], vec![1., -1., 0.]);
        assert_eq!(matrix(&r, "net")[0], vec![1., -1., 0.5]);
    }

    #[test]
    fn invalid_dimensions_nonfinite_inputs_and_inapplicable_bias_are_rejected() {
        let mut req = lecture_memory("hebbian_memory");
        req.t.pop();
        assert!(calculate(&req).is_err());
        let mut req = lecture_memory("outer_product");
        req.bias_enabled = true;
        assert!(calculate(&req).is_err());
        for s in [
            vec![],
            vec![vec![]],
            vec![vec![1.], vec![1., 2.]],
            vec![vec![f64::NAN]],
            vec![vec![1e51]],
            vec![vec![1.; 33]],
        ] {
            assert!(calculate(&Request {
                s,
                ..request("correlation")
            })
            .is_err());
        }
        let req = Request {
            w: vec![vec![1., 2.]],
            x: vec![vec![1., 2.]],
            ..request("recall")
        };
        assert!(calculate(&req).is_err());
        let mut req = Request {
            w: vec![vec![1., 2.]],
            x: vec![vec![1.]],
            bias_enabled: true,
            bias: vec![vec![1.]],
            ..request("recall")
        };
        assert!(calculate(&req).is_err());
        req.bias = vec![vec![0., 0.]];
        req.expected = vec![vec![1.]];
        assert!(calculate(&req).is_err());
    }
}
