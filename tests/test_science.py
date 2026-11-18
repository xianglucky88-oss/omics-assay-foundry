import math
import copy
import itertools
import pytest
from omics_assay_foundry.api import dispatch
from omics_assay_foundry.errors import DomainError

def test_mendelian_deterministic_parents(sample):
    result=dispatch('/pedigree',sample('pedigree'))
    assert result['posterior']['child']==pytest.approx([0,1,0])

def test_pedigree_ignores_absolute_likelihood_scale(sample):
    payload=sample('pedigree')
    for k in payload['likelihoods']:
        payload['likelihoods'][k]=[v*1e-250 for v in payload['likelihoods'][k]]
    result=dispatch('/pedigree',payload)
    assert result['posterior']['child']==pytest.approx([0,1,0])
    assert math.isfinite(result['log_evidence'])

def test_pedigree_uninformative_child_marginal_is_hardy_weinberg():
    result=dispatch('/pedigree',{'allele_frequency':0.2,'likelihoods':{role:[1,1,1] for role in ['father','mother','child']}})
    for probs in result['posterior'].values():
        assert probs==pytest.approx([0.64,0.32,0.04])

def test_pedigree_contradiction(sample):
    p=sample('pedigree');p['likelihoods']['child']=[1,0,0]
    with pytest.raises(DomainError) as error:dispatch('/pedigree',p)
    assert error.value.code=='incompatible_evidence'

def test_transcript_unique_fragment_length_correction():
    r=dispatch('/transcripts',{'effective_lengths':{'a':100,'b':200},'fragments':[['a'],['a'],['b']]})
    assert r['converged']
    assert [x['molecule_fraction'] for x in r['transcripts']]==pytest.approx([0.8,0.2])
    assert [x['expected_fragments'] for x in r['transcripts']]==pytest.approx([2,1])

def test_transcript_ambiguity_not_fixed_equal_split():
    r=dispatch('/transcripts',{'effective_lengths':{'a':100,'b':100},'fragments':[['a'],['a'],['b'],['a','b']]})
    assert [x['molecule_fraction'] for x in r['transcripts']]==pytest.approx([2/3,1/3])
    assert sum(x['expected_fragments'] for x in r['transcripts'])==pytest.approx(4)

def test_transcript_identifiability_is_reported():
    r=dispatch('/transcripts',{'effective_lengths':{'a':100,'b':100},'fragments':[['a','b']]})
    assert r['indistinguishable_groups']==[['a','b']]

def test_transcript_reordering(sample):
    p=sample('transcripts');a=dispatch('/transcripts',p);p['fragments'].reverse()
    assert [x['tpm'] for x in a['transcripts']]==pytest.approx([x['tpm'] for x in dispatch('/transcripts',p)['transcripts']])

def test_transcript_work_budget(sample):
    p=sample('transcripts');p['effective_lengths']['txA']=0
    with pytest.raises(DomainError):dispatch('/transcripts',p)

def test_contact_scaling_matches_analytic_two_by_two():
    result=dispatch('/contacts',{'matrix':[[4,1],[1,9]]})
    assert result['converged']
    assert result['weights']==pytest.approx([math.sqrt(3/14),math.sqrt(2/21)],rel=1e-7)
    assert result['matrix'][0]==pytest.approx([6/7,1/7],rel=1e-7)
    for row in result['matrix']:assert sum(row)==pytest.approx(1,abs=1e-8)

def test_contacts_single_cell():
    result=dispatch('/contacts',{'matrix':[[25]]})
    assert result['weights']==pytest.approx([0.2])
    assert result['matrix'][0]==pytest.approx([1])

@pytest.mark.parametrize('matrix',[[[1,0],[0,1]],[[1,2],[3,4]],[[1],[2]]])
def test_contact_invalid_matrices(matrix):
    with pytest.raises(DomainError):dispatch('/contacts',{'matrix':matrix})

def test_contact_budget_exhaustion_is_visible():
    result=dispatch('/contacts',{'matrix':[[4,1],[1,9]],'max_iterations':1,'tolerance':1e-12})
    assert not result['converged']
    assert result['residual']>1e-12

def test_wilson_known_interval():
    result=dispatch('/methylation',{'sites':[{'id':'s','methylated':5,'coverage':10}]})
    assert result['sites'][0]['interval']==pytest.approx([0.236593090512564,0.763406909487436])

def test_methylation_region_conservation_and_no_coverage(sample):
    p=sample('methylation');p['sites'].append({'id':'empty','methylated':0,'coverage':0})
    result=dispatch('/methylation',p)
    region=next(x for x in result['regions'] if x['id']=='r1')
    assert region['coverage']==20 and region['methylated']==12
    assert result['sites'][-1]['status']=='no_coverage'
    assert result['sites'][-1]['fraction'] is None

def test_methylation_no_silent_conversion_error(sample):
    p=sample('methylation');p['conversion_error']=0.01
    with pytest.raises(DomainError):dispatch('/methylation',p)

def test_jc69_pair_analytic_probability():
    tree={'children':[{'name':'a','length':0.1},{'name':'b','length':0.2}]}
    same=dispatch('/phylogeny',{'tree':tree,'sequences':{'a':'A','b':'A'}})
    different=dispatch('/phylogeny',{'tree':tree,'sequences':{'a':'A','b':'C'}})
    assert math.exp(same['log_likelihood'])==pytest.approx((0.25+0.75*math.exp(-4*0.3/3))/4)
    assert math.exp(different['log_likelihood'])==pytest.approx((0.25-0.25*math.exp(-4*0.3/3))/4)

def test_phylogeny_matches_independent_ancestor_sum():
    tree={'children':[{'name':'a','length':0.2},{'length':0.3,'children':[{'name':'b','length':0.4},{'name':'c','length':0.5}]}]}
    def transition(i,j,t):return 0.25+((0.75 if i==j else -0.25)*math.exp(-4*t/3))
    expected=sum(0.25*transition(root,0,0.2)*transition(root,ancestor,0.3)*transition(ancestor,1,0.4)*transition(ancestor,2,0.5)
                 for root,ancestor in itertools.product(range(4),repeat=2))
    actual=dispatch('/phylogeny',{'tree':tree,'sequences':{'a':'A','b':'C','c':'G'}})
    assert math.exp(actual['log_likelihood'])==pytest.approx(expected)

def test_zero_branch_incompatible_sequences_have_explicit_zero_probability():
    result=dispatch('/phylogeny',{'tree':{'children':[{'name':'a'},{'name':'b'}]},'sequences':{'a':'A','b':'C'}})
    assert result['status']=='zero_probability' and result['log_likelihood'] is None

def test_phylogeny_rejects_missing_leaf(sample):
    p=sample('phylogeny');del p['sequences']['b']
    with pytest.raises(DomainError):dispatch('/phylogeny',p)

def test_rna_two_base_partition_analytic():
    result=dispatch('/rna',{'sequence':'GC','minimum_loop':0})
    expected=math.exp(3)/(1+math.exp(3))
    assert result['structure_count']==2
    assert result['log_partition']==pytest.approx(math.log1p(math.exp(3)))
    assert result['pair_probabilities'][0][1]==pytest.approx(expected)

def test_rna_constraints_change_the_ensemble():
    required=dispatch('/rna',{'sequence':'GC','minimum_loop':0,'required_pairs':[[0,1]]})
    forbidden=dispatch('/rna',{'sequence':'GC','minimum_loop':0,'forbidden_pairs':[[0,1]]})
    assert required['pair_probabilities'][0][1]==1
    assert forbidden['pair_probabilities'][0][1]==0

def test_rna_empty_sequence_is_single_empty_structure():
    result=dispatch('/rna',{'sequence':''})
    assert result['structure_count']==1 and result['log_partition']==0

def test_rna_crossing_pairs_rejected():
    with pytest.raises(DomainError):dispatch('/rna',{'sequence':'GCGC','minimum_loop':0,'required_pairs':[[0,2],[1,3]]})

def test_rna_small_independent_matching_enumeration():
    seq='GCAU';pairs=[(0,1),(2,3),(0,3)]
    # Legal matchings: empty; any single allowed pair; the disjoint pair (0,1),(2,3).
    expected=1+math.exp(3)+math.exp(2)+math.exp(1)+math.exp(5)
    result=dispatch('/rna',{'sequence':seq,'minimum_loop':0})
    assert result['structure_count']==5
    assert result['log_partition']==pytest.approx(math.log(expected))
