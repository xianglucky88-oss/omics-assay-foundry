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
| `/contact-enrichment` | Sparse pair counts over declared bins with optional valid-bin list, disjoint distance strata and explicit relation queries; returns per-relation expected counts, log2 effects with Garwood intervals, one-sided Poisson p-values, Benjamini-Hochberg q-values for the enrichment and depletion families, the stratum background report and per-relation estimability statuses. | Model `sparse-multiplicative-marginal-distance-stratified-poisson-v1`: a listed pair (i, j) in stratum s is compared against mu = e_s * m_i * m_j with observed marginals m and a leave-one-out stratum scale e_s. The background population holds only listed pairs of valid bins inside converged positive-support components: masked bins and unlisted pairs are excluded (an unlisted pair is not a zero), listed zeros enter as true zeros. The sparse unit-marginal correction must exist and converge for a relation to be estimable; zero marginals, tree-structured or disconnected support, thin strata and exhausted budgets yield explicit inestimable statuses with null significance, never pseudocount-padded estimates. Relations beyond `max_relations` return `not_computed` and the result carries `complete: false`. |
| `/methylation` | Unique sites with methylated count, coverage and optional preassigned region; returns fractions and Wilson score intervals. | Zero conversion error. Coverage is a nonnegative integer. Regions pool independent Bernoulli trials under a common regional probability; heterogeneous or correlated sites are outside this interval model. Zero coverage returns a null fraction and [0, 1] interval. `confidence_z` is the normal quantile, not the confidence percentage. |
| `/phylogeny` | Rooted tree with nonnegative branch lengths and aligned ACGT strings; returns site and total log likelihood. | JC69 with unit expected substitution rate and uniform root prior; complete taxa, at most 32 leaves and 2000 sites. Root incoming branch length is unused. Impossible observations return `zero_probability` and null log likelihood. |
| `/rna` | Uppercase sequence plus optional loop minimum, required/forbidden pairs and beta; returns log partition, pair and unpaired probabilities. | Exact noncrossing ensemble, AU/GC/GU pairs, length at most 18, enumeration work at most 200000 generated combinations. A required crossing pair is rejected. Empty sequence is one empty structure. |

The RNA energy contract is **pair-additive-dimensionless-v1**: AU/UA=-2, GC/CG=-3 and GU/UG=-1, summed across pairs. A structure has weight exp(-beta * energy). Default beta is 1, and the default minimum number of intervening bases is 3. This is a deliberately restricted mathematical model; it is not a nearest-neighbor thermodynamic model or a clinical prediction. Energies have no implied kcal/mol conversion. Experimental free-energy parameters are not inferred.

## Responsibility boundaries

Each domain package separates input/model contracts, calculation and result assembly. `api.py` only selects a processor. `http_server.py` handles bounded JSON transport; it does not interpret scientific fields. `validation.py` checks common primitive types; domain rules stay in their domain package. No domain component imports another domain component.

## Numeric and operational limits

Inputs must be finite and satisfy the documented support range. Budget exhaustion, unsupported fields and incompatible evidence are distinct from valid zero results. All computation is local and stateless. Processing does not contact external services or write scientific inputs to disk. Output identifies its model so consumers can decide whether its assumptions apply. The service is a research data-processing component, not a medical device.
