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
| `/pedigree` | Two contracts. Legacy: alternative-allele frequency and nonnegative likelihood vectors for father, mother and child; returns genotype 0/1/2 posteriors and log evidence. Joint: `members` with ids, optional `ploidy` (1 or 2), `parents`, `source_parent` and per-member likelihood vectors; returns per-member posteriors, priors, MAP genotypes, information gain, log evidence and inference metadata. | Legacy is one complete diploid biallelic trio with independent Hardy-Weinberg parental priors. The joint contract evaluates the whole acyclic family as a single factor graph (founder priors, Mendelian transmission, genotype likelihoods), so unobserved parents and inbreeding loops are integrated over rather than averaged from split trios. Haploid members inherit from one declared source parent. Common ancestors are reported, not rejected; genuine parent cycles return `illegal_cycle`. Zero joint evidence is `incompatible_evidence`; an exceeded `max_states` budget is `budget_exhausted`. |
| `/transcripts` | Positive effective lengths and per-fragment compatible transcript identifiers; returns expected fragment allocations, molecule fractions, TPM, convergence and indistinguishable groups. | Fixed effective lengths, unbiased fragment compatibility model. Fragment allocation uses current molecule abundance; the molecule estimate accounts for effective length. No sampling uncertainty, fragment bias calibration or alignment. Completely unsupported transcripts receive zero abundance. Indistinguishable groups identify identical observed compatibility support, not every possible rank deficiency. |
| `/contacts` | Strictly positive, symmetric square contact matrix; returns weights, weighted matrix, unit-marginal residual and convergence. | Dense matrix up to 128 bins. Values lie in [1e-12, 1e12]. Structural zeros, masks and nonuniform targets are rejected. An iteration budget can return `converged: false`; this is not a successful balance certificate. |
| `/methylation` | Unique sites with methylated count, coverage and optional preassigned region; returns fractions and Wilson score intervals. | Zero conversion error. Coverage is a nonnegative integer. Regions pool independent Bernoulli trials under a common regional probability; heterogeneous or correlated sites are outside this interval model. Zero coverage returns a null fraction and [0, 1] interval. `confidence_z` is the normal quantile, not the confidence percentage. |
| `/phylogeny` | Rooted tree with nonnegative branch lengths and aligned ACGT strings; returns site and total log likelihood. | JC69 with unit expected substitution rate and uniform root prior; complete taxa, at most 32 leaves and 2000 sites. Root incoming branch length is unused. Impossible observations return `zero_probability` and null log likelihood. |
| `/rna` | Uppercase sequence plus optional loop minimum, required/forbidden pairs and beta; returns log partition, pair and unpaired probabilities. | Exact noncrossing ensemble, AU/GC/GU pairs, length at most 18, enumeration work at most 200000 generated combinations. A required crossing pair is rejected. Empty sequence is one empty structure. |

The RNA energy contract is **pair-additive-dimensionless-v1**: AU/UA=-2, GC/CG=-3 and GU/UG=-1, summed across pairs. A structure has weight exp(-beta * energy). Default beta is 1, and the default minimum number of intervening bases is 3. This is a deliberately restricted mathematical model; it is not a nearest-neighbor thermodynamic model or a clinical prediction. Energies have no implied kcal/mol conversion. Experimental free-energy parameters are not inferred.

## Joint pedigree model

The original trio request (`allele_frequency` plus `likelihoods.father/mother/child`) is unchanged. Sending `members` instead selects the joint contract. Each member declares an `id`; `ploidy` defaults to 2. Diploid members have zero or two parents, haploid members have one parent or two parents plus a `source_parent` naming the parent whose gamete is inherited. Members without a `likelihoods` vector are unobserved but remain random variables in the same model. A member reached through two lineage paths is a common ancestor (an inbreeding loop) and is listed in `common_ancestors`; only true directed parent cycles raise `illegal_cycle`.

Inference is exact over one factor graph. When the joint genotype space fits within `max_states` it is summed directly (`joint_enumeration`); otherwise reverse-topological variable elimination (`variable_elimination`) is used. Both engines implement the same factors and are cross-checked in the test suite, so results are never produced by splitting the family into trios and averaging. Every member returns its prior marginal alongside its posterior, a MAP genotype, posterior entropy and KL information gain, making explicit which conclusions come from assays (`support: observed`), which are inferred through relatives (`inferred`), and which still rest entirely on the model (`prior_only`, with an `insufficient_information` notice). Small families can be recomputed independently from the documented Hardy-Weinberg priors and Mendelian transmission probabilities; `log_evidence` is the log of the same normalizing constant. Likelihood vectors need not sum to one and their absolute scale does not affect posteriors.

## Responsibility boundaries

Each domain package separates input/model contracts, calculation and result assembly. `api.py` only selects a processor. `http_server.py` handles bounded JSON transport; it does not interpret scientific fields. `validation.py` checks common primitive types; domain rules stay in their domain package. No domain component imports another domain component.

## Numeric and operational limits

Inputs must be finite and satisfy the documented support range. Budget exhaustion, unsupported fields and incompatible evidence are distinct from valid zero results. All computation is local and stateless. Processing does not contact external services or write scientific inputs to disk. Output identifies its model so consumers can decide whether its assumptions apply. The service is a research data-processing component, not a medical device.
