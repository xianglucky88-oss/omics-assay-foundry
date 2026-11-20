from .errors import DomainError
from .validation import mapping
from .pedigree.posterior import process as pedigree
from .transcripts.abundance import process as transcripts
from .contacts.weighted_view import process as contacts
from .contacts.enrichment import process as contact_enrichment
from .methylation.regions import process as methylation
from .phylogeny.site_likelihoods import process as phylogeny
from .rna_constraints.pair_marginals import process as rna

ROUTES = {"/pedigree": pedigree, "/transcripts": transcripts, "/contacts": contacts,
          "/contact-enrichment": contact_enrichment,
          "/methylation": methylation, "/phylogeny": phylogeny, "/rna": rna}

def dispatch(path, payload):
    if path not in ROUTES:
        raise DomainError("unknown_endpoint", "Unknown endpoint")
    return ROUTES[path](mapping(payload))
