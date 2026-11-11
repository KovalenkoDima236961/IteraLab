# CLI review notes

## Source inspection limitation

`README.md` references `PRACTICAL_EXAMPLES_AUDIT.md`, but that file was absent
from this checkout when the CLI was implemented (including a repository-wide
filename search). It could not be inspected, and no replacement practical audit
has been invented. The existing Rust implementations, Python adapters,
source-labelled presets, README, and numerical/practical integration tests
were used as the authoritative implementation and reproducible examples.

## Existing mathematical conventions retained

No numerical module or Python calculation adapter was changed. The CLI shares
the existing Rust dispatcher and result serializers; changes to `main.rs`
only route the existing output through a caller-supplied writer and select the
explicit CLI entry point when arguments are present.

The Lecture 08 correlation worksheet exposes a literal lecture quantity
`lecture_alpha_upper = lambda_max / 2` as well as separate mean-LMS analysis.
That literal quantity must not be mistaken for the reciprocal-curvature rate
or a sequential LMS guarantee. It is retained unchanged and is not used by CLI
parameter selection. CLI ADALINE selects its fixed rate from individual
augmented sample norms, including the trained bias, with assumptions and
limitations documented in `CLI.md`.

Nelder–Mead slide branches and standard branches are intentionally distinct.
Function-spread stopping can accept a geometrically large flat simplex.
Perceptron bipolar sign and bipolar error updates intentionally differ by the
error multiplier, and Hebbian completion intentionally does not imply correct
classification. Newton's gradient stopping alone does not establish a minimum.
These are existing, tested behaviors, not changes introduced by the CLI.

No new suspected numerical defect was corrected or folded into this work.

## Supplied steepest-descent lecture images

The images supplied after implementation describe an optimized step each
iteration, `alpha_k = argmin_{alpha>0} f(X_k-alpha*gradient(f)(X_k))`. The CLI
exposes the existing corresponding direction/search path as
`steepest-line-search`. The new `1/lambda_max(H)` parameter selector belongs
only to the separate fixed-gradient variant; it is not that lecture rule.

The existing optimized backend uses a bracketed directional-derivative root
search, accepts a non-increasing stationary trial within its relative residual
tolerance, and stops at gradient norm `<= tolerance` or the update budget. It
does not perform an explicit second-derivative minimum test or implement a
Golden Section fallback. The images show strict `<` gradient stopping and
also list displacement/function-change stopping criteria; those are not
selectable in this backend. These differences are recorded for review without
changing the tested numerical implementation.
