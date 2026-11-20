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
| `/transcripts` | Positive effective lengths and per-fragment compatible transcript identifiers; returns expected fragment allocations, molecule fractions, TPM, convergence and indistinguishable groups. | Fixed effective lengths, unbiased fragment compatibility model. Fragment allocation uses current molecule abundance; the molecule estimate accounts for effective length. No sampling uncertainty, fragment bias calibration or alignment. Completely unsupported transcripts receive zero abundance. Indistinguishable groups identify identical observed compatibility support, not every possible rank deficiency. |
| `/contacts` | Strictly positive, symmetric square contact matrix; returns weights, weighted matrix, unit-marginal residual and convergence. | Dense matrix up to 128 bins. Values lie in [1e-12, 1e12]. Structural zeros, masks and nonuniform targets are rejected. An iteration budget can return `converged: false`; this is not a successful balance certificate. |
| `/methylation` | Unique sites with methylated count, coverage and optional preassigned region; returns fractions and Wilson score intervals. | Zero conversion error. Coverage is a nonnegative integer. Regions pool independent Bernoulli trials under a common regional probability; heterogeneous or correlated sites are outside this interval model. Zero coverage returns a null fraction and [0, 1] interval. `confidence_z` is the normal quantile, not the confidence percentage. |
| `/phylogeny` | Rooted tree with nonnegative branch lengths and aligned ACGT strings; returns site and total log likelihood. | JC69 with unit expected substitution rate and uniform root prior; complete taxa, at most 32 leaves and 2000 sites. Root incoming branch length is unused. Impossible observations return `zero_probability` and null log likelihood. |
| `/phylogeny/evidence` | Finite candidate trees, aligned sequences (IUPAC plus missing), optional site partitions, shared-parameter declarations, GTR and discrete rate mixtures; returns per-partition and per-site log likelihoods, candidate deltas, site-resampling stability, diagnostics and independent verification. | Compares only the supplied candidates on one observation population and one scale; no tree search or branch re-estimation. Taxa missing from a candidate are rejected rather than skipped; ambiguous or missing characters widen the state set; every column belongs to exactly one partition. See below. |
| `/rna` | Uppercase sequence plus optional loop minimum, required/forbidden pairs and beta; returns log partition, pair and unpaired probabilities. | Exact noncrossing ensemble, AU/GC/GU pairs, length at most 18, enumeration work at most 200000 generated combinations. A required crossing pair is rejected. Empty sequence is one empty structure. |

The RNA energy contract is **pair-additive-dimensionless-v1**: AU/UA=-2, GC/CG=-3 and GU/UG=-1, summed across pairs. A structure has weight exp(-beta * energy). Default beta is 1, and the default minimum number of intervening bases is 3. This is a deliberately restricted mathematical model; it is not a nearest-neighbor thermodynamic model or a clinical prediction. Energies have no implied kcal/mol conversion. Experimental free-energy parameters are not inferred.

### Phylogeny evidence comparison

`POST /phylogeny/evidence` compares a finite set of supplied candidate trees on one multi-sequence observation. It does not search for trees and does not re-estimate branch lengths. Request fields:

| Field | Meaning |
|---|---|
| `sequences` | Aligned sequences, 2 to 32 taxa, 1 to 2000 columns. Complete `ACGT` is not required: IUPAC ambiguity codes expand to the states they admit, and `-`/`?` mean missing data (all four states). Unsupported characters are rejected. |
| `candidates` | 1 to 16 entries, each `{id?, name?, tree, model?}`. Every candidate must contain every observed taxon exactly once; a different taxon set is rejected with `candidate_coverage`, because it would compare a different observation population. |
| `partitions` | Optional. If absent, one partition `all` covers every column. Otherwise each entry is `{id, sites, taxa?, model?}`. `sites` is a list of indices or a half-open `{"start", "end"}` range. Every column must belong to exactly one partition. A partition may observe only a subset of taxa. |
| `shared_parameters` | Optional list from `substitution`, `frequencies`, `exchangeabilities`, `rate_categories`. A declared parameter must be equal across every effective model; otherwise the request fails with `shared_parameter_violation`. |
| `resampling` | Optional `{replicates, seed}`. Default 200 replicates and seed 0. `replicates: 0` returns an explicit `no_replicates_requested` state. |
| `budget` | Optional `{max_site_likelihoods, max_resampling_work}`. The former bounds `candidates x sites`; if it is too small, every candidate is marked `budget_exhausted` rather than partially evaluated. |
| `distinguishability_threshold` | Log-likelihood gap below which the best and another candidate are reported as indistinguishable. Default 2. |
| `site_dominance_threshold` | Share of the positive leading difference concentrated in one column that triggers a `site_dominated` caveat. Default 0.5. |

**Observations are not silently reduced.** Every observed column is evaluated for every candidate. An ambiguous or missing character widens that leaf's state set; it never deletes the column. A taxon excluded from a partition is marginalized exactly (its conditional likelihood vector is all ones), so the column still counts. A candidate whose tree omits a taxon is rejected instead of being allowed to skip that taxon's difficult columns. A column assigned to no partition is rejected with `partition_coverage`.

**One scale.** Branch lengths are expected substitutions per site. Each substitution model is normalized so its expected rate is exactly one, and explicit rate categories must have weighted mean rate one (the service rejects them otherwise rather than silently rescaling). Model descriptions are echoed per partition so a consumer can check that candidates were compared on the same footing. `shared_parameters` makes the intent explicit and is enforced.

**Models.** `JC69` (uniform frequencies, equal exchangeabilities, the original single-tree model) or `GTR` (requires both a symmetric positive exchangeability matrix and frequencies summing to one). `rate_categories` accepts either explicit `rates`/`weights` or `{gamma_shape, categories}` for the equal-probability discrete-gamma mixture (Yang 1994) with mean rate one.

**Results.** For each candidate: per-partition and per-site log likelihoods, per-partition totals and status; `delta_from_best` totals; per-partition deltas that sum to the total; and an overall status of `ok`, `zero_probability`, `numeric_instability` or `budget_exhausted`. A column with probability exactly zero is reported in `zero_probability_sites` and is not converted into a dropped column, a zero total, or a failed candidate that "wins" by exclusion. A column contributing exactly zero log likelihood (for example all-missing data) is reported separately as `uninformative_sites`: no information is not the same statistical statement as zero probability.

**Resampling.** `resampling` performs a nonparametric bootstrap over columns. Each replicate resamples column indices once and applies the same indices to every candidate and partition. `bootstrap_support_fraction` is the fraction of comparable replicates in which a candidate ranked strictly highest. Replicate totals within a round-off tolerance are counted as ties, so floating-point summation order never decides a winner. It is **not** a posterior probability and **not** the probability that a tree is true; the field `support_interpretation` says so in the response. A column that is impossible for a candidate contributes `-inf`, so that candidate loses any replicate containing it; it is never excused from the comparison. Replicates containing a numerically indeterminate column are counted as incomparable. For small problems (few columns) the exact multinomial resampling distribution is also computed and compared with the Monte Carlo estimate; for larger problems `exact_reference` reports `skipped_too_large`.

**Diagnostics and explicable states.** `site_dominance` reports when a few columns carry most of the leading difference. `diagnostics` reports minimum transition probability, row-sum error, clamped entries from round-off, zero-length branch counts and maximum depth. Deep trees that produce materially negative raw transition probabilities are flagged in `caveats`. A budget that cannot cover the requested work yields `budget_exhausted`, not a zero result. A candidate set that the data cannot separate yields `indistinguishable` rather than an arbitrary winner. `verification` recomputes small candidates by direct enumeration of internal node states and reports whether it agrees.

## Responsibility boundaries

Each domain package separates input/model contracts, calculation and result assembly. `api.py` only selects a processor. `http_server.py` handles bounded JSON transport; it does not interpret scientific fields. `validation.py` checks common primitive types; domain rules stay in their domain package. No domain component imports another domain component.

## Numeric and operational limits

Inputs must be finite and satisfy the documented support range. Budget exhaustion, unsupported fields and incompatible evidence are distinct from valid zero results. All computation is local and stateless. Processing does not contact external services or write scientific inputs to disk. Output identifies its model so consumers can decide whether its assumptions apply. The service is a research data-processing component, not a medical device.
