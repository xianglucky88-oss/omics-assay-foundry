# Omics Assay Foundry

A library and local HTTP service that converts defined molecular assay inputs into reproducible numeric datasets. The six components are independent. Runtime dependencies: Python 3.11+ standard library. Tests: pytest 8.x or 9.x. Wheel build backend: setuptools 68+.

## Run

```bash
python -m pip install .
python -m omics_assay_foundry --port 9840
```

`GET /health` lists available endpoints. All processing endpoints accept JSON via POST and return `{"result": ...}`. Scientific input errors return HTTP 422 with an error code. Malformed JSON returns 400. Request bodies are limited to 2 MiB. The default listener is loopback. Public deployment, authentication and reverse-proxy configuration are outside this local service.

```bash
curl -H 'Content-Type: application/json' --data-binary @examples/pedigree.json http://127.0.0.1:9840/pedigree
PYTHONPATH=src python -m pytest
```

Every endpoint has a complete request in `examples/`. As a library, use `omics_assay_foundry.api.dispatch(path, payload)` or the component's `process` function. Unknown options are rejected, rather than silently ignored.

## Processing contracts

| Endpoint | Inputs and output | Supported scope |
|---|---|---|
| `/pedigree` | Alternative-allele frequency and nonnegative likelihood vectors for father, mother and child; returns genotype 0/1/2 posteriors and log evidence. | One complete diploid biallelic trio, independent parental Hardy-Weinberg priors, Mendelian transmission, no mutations. Likelihood vectors need not sum to one. Zero model evidence is an explicit error. |
| `/transcripts` | Legacy: positive effective lengths and per-fragment compatible transcript identifiers. Extended: annotated transcript lengths plus an observed fragment-length distribution, relative fragment-to-transcript bias weights, replicate observation counts and a resampling configuration. Returns expected allocations, molecule fractions, TPM, effective lengths, per-transcript support status, identifiability groups, resampling intervals and their usability. | See the transcript quantification contract below. |
| `/contacts` | Strictly positive, symmetric square contact matrix; returns weights, weighted matrix, unit-marginal residual and convergence. | Dense matrix up to 128 bins. Values lie in [1e-12, 1e12]. Structural zeros, masks and nonuniform targets are rejected. An iteration budget can return `converged: false`; this is not a successful balance certificate. |
| `/methylation` | Unique sites with methylated count, coverage and optional preassigned region; returns fractions and Wilson score intervals. | Zero conversion error. Coverage is a nonnegative integer. Regions pool independent Bernoulli trials under a common regional probability; heterogeneous or correlated sites are outside this interval model. Zero coverage returns a null fraction and [0, 1] interval. `confidence_z` is the normal quantile, not the confidence percentage. |
| `/phylogeny` | Rooted tree with nonnegative branch lengths and aligned ACGT strings; returns site and total log likelihood. | JC69 with unit expected substitution rate and uniform root prior; complete taxa, at most 32 leaves and 2000 sites. Root incoming branch length is unused. Impossible observations return `zero_probability` and null log likelihood. |
| `/rna` | Uppercase sequence plus optional loop minimum, required/forbidden pairs and beta; returns log partition, pair and unpaired probabilities. | Exact noncrossing ensemble, AU/GC/GU pairs, length at most 18, enumeration work at most 200000 generated combinations. A required crossing pair is rejected. Empty sequence is one empty structure. |

The RNA energy contract is **pair-additive-dimensionless-v1**: AU/UA=-2, GC/CG=-3 and GU/UG=-1, summed across pairs. A structure has weight exp(-beta * energy). Default beta is 1, and the default minimum number of intervening bases is 3. This is a deliberately restricted mathematical model; it is not a nearest-neighbor thermodynamic model or a clinical prediction. Energies have no implied kcal/mol conversion. Experimental free-energy parameters are not inferred.

## Transcript quantification contract

`POST /transcripts` has two shapes. A request whose only fields are `effective_lengths`, `fragments`, `max_iterations` and `tolerance` keeps the original **fixed-effective-length-unbiased-fragments** model and output. Any of the measurement fields below switches to the extended **fragment-length-and-bias-weighted** model; the legacy output fields are not silently reinterpreted.

### Measurement fields

| Field | Meaning |
|---|---|
| `transcript_lengths` | Annotated transcript lengths `T_i`. |
| `fragment_lengths` | One observed fragment length per entry of `fragments`. |
| `fragment_length_distribution` | Observed length distribution instead of per-fragment lengths: a `{length: weight}` histogram, `{"lengths": ..., "weights"\|"counts": ...}`, `[length, weight]` pairs, a plain list of lengths, or a normal summary `{"mean": ..., "sd": ...}` discretized over whole bases. |
| `fragment_bias` | One relative weight object or vector per fragment. |
| `bias_weights` | One relative weight per transcript, applied to all of its fragments. |
| `observation_counts` | Replicate observations per fragment, or one integer for all fragments. |
| `resamples`, `seed`, `chunk_size`, `bootstrap_work_limit` | Resampling configuration; the same keys are accepted nested under `random`, `resampling` or `bootstrap`. `bootstrap_work_limit` caps resampling work (default 50 million unit-updates). |
| `low_abundance_tpm` | TPM threshold below which a supported transcript is reported as `low_abundance` (default 1). |

Fragments may be objects `{"compatible": [...], "length": ..., "bias": {...}, "count": ...}` instead of name lists with parallel arrays; the forms cannot be mixed. Recognized aliases resolve onto a canonical field, and conflicting values for one field are rejected.

### Effective length and normalization

The effective length is `L_i = E_l[A_i(l)]`, the mean number of start positions available to a fragment drawn from the observed length distribution, with `A_i(l) = max(T_i - l + 1, 0)`. It is derived from `transcript_lengths` and the fragment-length distribution; a supplied `effective_lengths` is checked against it rather than preferred, and a disagreement is an error. The same `L_i` is used by the allocation, the read-to-molecule conversion and the reported `effective_length`, so there is one convention on both sides of the request. TPM is `molecule_fraction * 1e6`, and molecule fractions sum to one; expected fragment allocations sum to the total observation count.

### Model

A fragment observation is allocated over its compatible transcripts in proportion to the current molecule fraction times the declared relative bias `b_ji` (default 1). Conditional on the fragment's declared member set, its length density and uniform start position are shared by all members and cancel, so the observed fragment length does not reweight the members; it enters abundance through each transcript's effective length `L_i`. Only ratios of `b` within one fragment matter, so a global rescaling cancels, and with no bias the allocation is the legacy unbiased one.

The M-step divides allocated fragments by the effective yield `D_i = L_i * B_i`, where `B_i` is the mean declared bias over the observations that support transcript `i` (`B_i = 1` when none is declared). This is what makes the measurement conditions affect abundance: a condition that raises detection (`b > 1`) raises `D_i` and therefore lowers the molecules implied by the same fragment count. With no declared bias `D_i` is exactly the reported effective length, so an extended request carrying no new measurement field reproduces the legacy result. Observation counts weight allocation and are resampled as independent Poisson(1) units. This is a relative-abundance model with declared propensity weights, not an alignment, bias-calibration or differential-expression procedure; replicate counts describe measurement effort, not biological replication.

### Reporting

* `effective_length` and `effective_yield` are reported per transcript. With no declared bias the yield equals the effective length; a declared detection bias multiplies it (see the model note above).
* `converged` and `convergence_meaning`: convergence only means the iteration stopped moving. It does **not** mean every transcript is individually identified.
* `support_status` per transcript, distinct in `status_counts` and `comments`: `identified`, `low_abundance` (below `low_abundance_tpm`), `zero_support` (no observed fragment supports it; a model zero, not measured absence), and `indistinguishable`.
* `indistinguishable_groups` and `group_details`: transcripts whose observed support and propensity columns are proportional are identified only through `sum(coefficient * fraction)`. Group details report the coefficients, that identified combination and its interval; the individual split is arbitrary, so members get no individual interval.
* `uncertainty`: `method` is a Poisson observation bootstrap with `level` 0.95. `status` is `ok`, `not_requested`, `budget_exceeded` or `insufficient_replicates`; only `ok` sets `usable: true`. `resamples` requires an explicit integer `seed`; an explicit seed with no `resamples` selects the default 200.
* `usable_for_quantitative_conclusions` is false when the point estimate did not converge or when a requested interval could not be formed.

### Reproducibility

Draws are keyed by `(seed, replicate, global observation-unit index)` with a counter-based generator, and allocation accumulates in request order. The same explicit `seed` and fragment list therefore produce identical output for any `chunk_size`: a job may cut the same fragments into differently sized chunks without changing the numbers. Different seeds change the intervals but not the point estimates.

A small worked request is in `examples/transcripts_weighted.json`.

## Responsibility boundaries

Each domain package separates input/model contracts, calculation and result assembly. `api.py` only selects a processor. `http_server.py` handles bounded JSON transport; it does not interpret scientific fields. `validation.py` checks common primitive types; domain rules stay in their domain package. No domain component imports another domain component.

## Numeric and operational limits

Inputs must be finite and satisfy the documented support range. Budget exhaustion, unsupported fields and incompatible evidence are distinct from valid zero results. All computation is local and stateless. Processing does not contact external services or write scientific inputs to disk. Output identifies its model so consumers can decide whether its assumptions apply. The service is a research data-processing component, not a medical device.
