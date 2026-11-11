//! Infix input adapter. Evaluation and differentiation stay in optimization.rs.
use crate::optimization::{compile_variables, Instruction};

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(f64),
    Name(String),
    Op(char),
    End,
}

struct Parser {
    tokens: Vec<Token>,
    at: usize,
    output: Vec<String>,
    dimensions: usize,
}

pub fn compile(source: &str, dimensions: usize) -> Result<String, String> {
    if source.is_empty() || source.len() > 1000 {
        return Err("expression must contain 1–1000 bytes".into());
    }
    let bytes = source.as_bytes();
    let mut i = 0;
    let mut tokens = Vec::new();
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || c == '.' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            if i < bytes.len() && matches!(bytes[i], b'e' | b'E') {
                i += 1;
                if i < bytes.len() && matches!(bytes[i], b'+' | b'-') {
                    i += 1;
                }
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let value: f64 = source[start..i]
                .parse()
                .map_err(|_| "invalid expression number")?;
            if !value.is_finite() || value.abs() > 1e100 {
                return Err("expression constant exceeds 1e100".into());
            }
            tokens.push(Token::Number(value));
        } else if c.is_ascii_alphabetic() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
                i += 1;
            }
            tokens.push(Token::Name(source[start..i].into()));
        } else if "+-*/^()".contains(c) {
            i += 1;
            if c == '*' && i < bytes.len() && bytes[i] == b'*' {
                i += 1;
                tokens.push(Token::Op('^'));
            } else {
                tokens.push(Token::Op(c));
            }
        } else {
            return Err(format!("invalid expression character at byte {}", i + 1));
        }
    }
    tokens.push(Token::End);
    let mut p = Parser {
        tokens,
        at: 0,
        output: Vec::new(),
        dimensions,
    };
    p.expression(0, 0)?;
    if p.peek() != &Token::End {
        return Err("unexpected expression token; use explicit multiplication (2*x)".into());
    }
    let result = p.output.join(" ");
    compile_variables(&result, dimensions == 2)?;
    Ok(result)
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.at]
    }
    fn expression(&mut self, minimum: u8, depth: usize) -> Result<(), String> {
        if depth > 30 {
            return Err("expression nesting exceeds 30".into());
        }
        let first = self.peek().clone();
        self.at += 1;
        match first {
            Token::Number(v) => self.output.push(format!("c:{v:?}")),
            Token::Name(name) => match name.as_str() {
                "x" => self.output.push(name),
                "y" if self.dimensions == 2 => self.output.push(name),
                "pi" => self.output.push(format!("c:{:?}", std::f64::consts::PI)),
                "e" => self.output.push(format!("c:{:?}", std::f64::consts::E)),
                "sin" | "cos" | "tan" | "exp" | "ln" | "log" | "sqrt" => {
                    if self.peek() != &Token::Op('(') {
                        return Err(format!("{name} requires parentheses"));
                    }
                    self.at += 1;
                    self.expression(0, depth + 1)?;
                    self.close()?;
                    self.output
                        .push(if name == "log" { "ln".into() } else { name });
                }
                _ => {
                    return Err(format!(
                        "unknown expression name '{name}' in {}D",
                        self.dimensions
                    ))
                }
            },
            Token::Op('(') => {
                self.expression(0, depth + 1)?;
                self.close()?;
            }
            Token::Op(sign @ ('+' | '-')) => {
                // Power binds more tightly than unary signs, as in the Python adapter.
                self.expression(3, depth + 1)?;
                if sign == '-' {
                    self.output.push("neg".into());
                }
            }
            _ => {
                return Err(
                    "expected a number, variable, function or parenthesized expression".into(),
                )
            }
        }
        loop {
            let (op, precedence, right) = match self.peek() {
                Token::Op('+') => ("add", 1, false),
                Token::Op('-') => ("sub", 1, false),
                Token::Op('*') => ("mul", 2, false),
                Token::Op('/') => ("div", 2, false),
                Token::Op('^') => ("pow", 4, true),
                _ => break,
            };
            if precedence < minimum {
                break;
            }
            self.at += 1;
            self.expression(precedence + u8::from(!right), depth + 1)?;
            self.output.push(op.into());
            if self.output.len() > 200 {
                return Err("expression exceeds 200 instructions".into());
            }
        }
        Ok(())
    }
    fn close(&mut self) -> Result<(), String> {
        if self.peek() != &Token::Op(')') {
            return Err("missing closing parenthesis".into());
        }
        self.at += 1;
        Ok(())
    }
}

// Conservative structural certificate for constant curvature; never infer a global
// bound from a Hessian sampled at one point. This is parameter selection only.
type Polynomial = [[f64; 3]; 3];
fn constant(v: f64) -> Polynomial {
    let mut p = [[0.; 3]; 3];
    p[0][0] = v;
    p
}
fn scalar(p: &Polynomial) -> Option<f64> {
    (p.iter()
        .enumerate()
        .all(|(i, row)| row.iter().enumerate().all(|(j, &v)| i + j == 0 || v == 0.)))
    .then_some(p[0][0])
}
fn product(a: Polynomial, b: Polynomial) -> Option<Polynomial> {
    let mut p = constant(0.);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    if a[i][j] != 0. && b[k][l] != 0. && i + j + k + l > 2 {
                        return None;
                    }
                    let term = a[i][j] * b[k][l];
                    if term != 0. {
                        if i + j + k + l > 2 {
                            return None;
                        }
                        p[i + k][j + l] += term;
                    }
                }
            }
        }
    }
    Some(p)
}

pub fn quadratic_rate(source: &str, dimensions: usize) -> Result<(f64, String), String> {
    let unsupported = "calculated mode requires a structurally quadratic, positive-definite objective; use provided mode for other objectives";
    let program = compile_variables(source, dimensions == 2)?;
    let mut stack: Vec<Polynomial> = Vec::new();
    for instruction in program {
        let p = match instruction {
            Instruction::Constant(v) => constant(v),
            Instruction::X => {
                let mut p = constant(0.);
                p[1][0] = 1.;
                p
            }
            Instruction::Y => {
                let mut p = constant(0.);
                p[0][1] = 1.;
                p
            }
            Instruction::Unary(op) if op == "neg" => {
                stack.pop().ok_or(unsupported)?.map(|row| row.map(|v| -v))
            }
            Instruction::Binary(op) => {
                let b = stack.pop().ok_or(unsupported)?;
                let a = stack.pop().ok_or(unsupported)?;
                match op.as_str() {
                    "add" | "sub" => std::array::from_fn(|i| {
                        std::array::from_fn(|j| {
                            a[i][j] + if op == "add" { b[i][j] } else { -b[i][j] }
                        })
                    }),
                    "mul" => product(a, b).ok_or(unsupported)?,
                    "div" => {
                        let d = scalar(&b).filter(|v| *v != 0.).ok_or(unsupported)?;
                        a.map(|row| row.map(|v| v / d))
                    }
                    "pow" => match scalar(&b) {
                        Some(0.) => constant(1.),
                        Some(1.) => a,
                        Some(2.) => product(a, a).ok_or(unsupported)?,
                        _ => return Err(unsupported.into()),
                    },
                    _ => return Err(unsupported.into()),
                }
            }
            _ => return Err(unsupported.into()),
        };
        if p.iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > 1e100)
        {
            return Err(unsupported.into());
        }
        stack.push(p);
    }
    let p = stack.pop().ok_or(unsupported)?;
    let (a, b, c) = (2. * p[2][0], p[1][1], 2. * p[0][2]);
    let (small, large) = if dimensions == 1 {
        (a, a)
    } else {
        let scale = a.abs().max(b.abs()).max(c.abs());
        if scale == 0. || !scale.is_finite() {
            return Err(unsupported.into());
        }
        let (aa, bb, cc) = (a / scale, b / scale, c / scale);
        let high = 0.5 * (aa + cc + (aa - cc).hypot(2. * bb));
        if high <= 0. {
            return Err(unsupported.into());
        }
        ((aa * cc - bb * bb) / high * scale, high * scale)
    };
    let rate = 1. / large;
    if small <= 0.
        || !small.is_finite()
        || !large.is_finite()
        || !rate.is_finite()
        || rate <= 0.
        || rate > 1e50
    {
        return Err(unsupported.into());
    }
    Ok((rate, format!("rule = 1 / lambda_max(H), calculated once and fixed\nassumption = structurally certified positive-definite quadratic\nH_xx = {a:.17e}\nH_xy = {b:.17e}\nH_yy = {c:.17e}\nlambda_min = {small:.17e}\nlambda_max = {large:.17e}\n")))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn precedence_matches_the_existing_python_adapter() {
        for (source, rpn) in [
            ("-x^2", "x c:2.0 pow neg"),
            ("2^-2", "c:2.0 c:2.0 neg pow"),
            ("x**2**3", "x c:2.0 c:3.0 pow pow"),
            ("log(1e-3)+y", "c:0.001 ln y add"),
            ("2*x/(1+y)", "c:2.0 x mul c:1.0 y add div"),
        ] {
            assert_eq!(compile(source, 2).unwrap(), rpn);
        }
    }
    #[test]
    fn rejects_invalid_syntax_without_panicking() {
        for s in [
            "", "x+", "(x", "sin()", "x)", "2x", "y", "x[0]", "foo(x)", "1e", "NaN",
        ] {
            assert!(compile(s, 1).is_err(), "{s}");
        }
    }
    #[test]
    fn quadratic_certificate_and_actual_rate() {
        assert_eq!(
            quadratic_rate(&compile("(x-2)^2+1", 1).unwrap(), 1)
                .unwrap()
                .0,
            0.5
        );
        assert_eq!(
            quadratic_rate(&compile("x^2+x*y+y^2", 2).unwrap(), 2)
                .unwrap()
                .0,
            1. / 3.
        );
        for s in [
            "exp(x)+x^2",
            "x^4",
            "-x^2",
            "x^2-y^2",
            "x^2",
            "(x*y)^2",
            "x/y",
        ] {
            assert!(quadratic_rate(&compile(s, 2).unwrap(), 2).is_err(), "{s}");
        }
    }
}
