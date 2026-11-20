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

Every endpoint has a complete request in `examples/`. `/methylation` also ships `examples/methylation_controls.json`, which carries per-batch, per-strand conversion controls. As a library, use `omics_assay_foundry.api.dispatch(path, payload)` or the component's `process` function. Unknown options are rejected, rather than silently ignored.

## Processing contracts

| Endpoint | Inputs and output | Supported scope |
|---|---|---|
| `/pedigree` | Alternative-allele frequency and nonnegative likelihood vectors for father, mother and child; returns genotype 0/1/2 posteriors and log evidence. | One complete diploid biallelic trio, independent parental Hardy-Weinberg priors, Mendelian transmission, no mutations. Likelihood vectors need not sum to one. Zero model evidence is an explicit error. |
| `/transcripts` | Positive effective lengths and per-fragment compatible transcript identifiers; returns expected fragment allocations, molecule fractions, TPM, convergence and indistinguishable groups. | Fixed effective lengths, unbiased fragment compatibility model. Fragment allocation uses current molecule abundance; the molecule estimate accounts for effective length. No sampling uncertainty, fragment bias calibration or alignment. Completely unsupported transcripts receive zero abundance. Indistinguishable groups identify identical observed compatibility support, not every possible rank deficiency. |
| `/contacts` | Strictly positive, symmetric square contact matrix; returns weights, weighted matrix, unit-marginal residual and convergence. | Dense matrix up to 128 bins. Values lie in [1e-12, 1e12]. Structural zeros, masks and nonuniform targets are rejected. An iteration budget can return `converged: false`; this is not a successful balance certificate. |
| `/methylation` | Unique sites with methylated count, coverage and optional preassigned region, plus optional spike-in conversion controls; returns fractions and intervals. | Without `controls`, zero conversion error: independent Bernoulli trials per region, Wilson score interval. With `controls`, each batch/strand pair carries a fully methylated and a fully unmethylated count; the biological fraction satisfies `q = u + p(1 - c - u)` with Beta posteriors on `c` and `u` and a uniform prior on `p`, reported as an HPD credible interval. Regions keep the homogeneous-site assumption and pool counts under one regional probability; sites sharing a control form one binomial pool, so the control posterior is integrated once per region and never inflates the observed coverage. Controls with no coverage, with indistinguishable fully methylated and unmethylated rates, or absent for a site's strand yield an unidentifiable cell (`no_control_coverage`, `control_unresolved`, `missing_control`, `ambiguous_control`) with a [0, 1] interval rather than a rejection. Fully unmethylated or fully methylated sites report `near_boundary`. Every calibrated cell and region names its control source. Coverage is a nonnegative integer. Zero coverage returns a null fraction and [0, 1] interval. `confidence_z` is the normal quantile, not the confidence percentage. |
| `/phylogeny` | Rooted tree with nonnegative branch lengths and aligned ACGT strings; returns site and total log likelihood. | JC69 with unit expected substitution rate and uniform root prior; complete taxa, at most 32 leaves and 2000 sites. Root incoming branch length is unused. Impossible observations return `zero_probability` and null log likelihood. |
| `/rna` | Uppercase sequence plus optional loop minimum, required/forbidden pairs and beta; returns log partition, pair and unpaired probabilities. | Exact noncrossing ensemble, AU/GC/GU pairs, length at most 18, enumeration work at most 200000 generated combinations. A required crossing pair is rejected. Empty sequence is one empty structure. |

The RNA energy contract is **pair-additive-dimensionless-v1**: AU/UA=-2, GC/CG=-3 and GU/UG=-1, summed across pairs. A structure has weight exp(-beta * energy). Default beta is 1, and the default minimum number of intervening bases is 3. This is a deliberately restricted mathematical model; it is not a nearest-neighbor thermodynamic model or a clinical prediction. Energies have no implied kcal/mol conversion. Experimental free-energy parameters are not inferred.

The methylation conversion model is **controls-v1**. A `controls` array holds records of the form `{"id", "batch", "strand", "fully_methylated": {"methylated", "coverage"}, "fully_unmethylated": {"methylated", "coverage"}}`; a batch→strand→record mapping is also accepted. A site selects a control by `control` id, by matching `batch`/`strand`, or implicitly when exactly one control fits; an explicit id that is not in the registry is an `unknown_control` error. Requests without `controls` keep the original zero-conversion-error model and Wilson intervals, and do not accept the site selection fields. `conversion_error` must be zero in either mode; a nonzero value is rejected instead of being silently ignored, because conversion bias is now estimated from the control measurements. The reported interval is a highest-posterior-density credible interval for the biological fraction; low control coverage widens it, no control coverage or indistinguishable controls make it unidentifiable, and the region level integrates each control's posterior exactly once. Interval coverage under a known conversion setting is exercised by repeated-measurement simulation in `tests/test_methylation_controls.py`.

## Responsibility boundaries

Each domain package separates input/model contracts, calculation and result assembly. `api.py` only selects a processor. `http_server.py` handles bounded JSON transport; it does not interpret scientific fields. `validation.py` checks common primitive types; domain rules stay in their domain package. No domain component imports another domain component.

## Numeric and operational limits

Inputs must be finite and satisfy the documented support range. Budget exhaustion, unsupported fields and incompatible evidence are distinct from valid zero results. All computation is local and stateless. Processing does not contact external services or write scientific inputs to disk. Output identifies its model so consumers can decide whether its assumptions apply. The service is a research data-processing component, not a medical device.
