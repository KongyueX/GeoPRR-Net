"""Check revision cache ordering and actual feature/target separation on small fixtures."""
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.revision_cache import collect_revision_cache, effective_availability
from experiments.resnet18_direct_progress import DirectSample
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS


class Reader(torch.nn.Module):
    def forward_experts(self, raw, normalized, support, *, sarn_active, raw_to_sarn_homography):
        n=len(raw)
        candidates=torch.tensor([[.2,.4,.6]]).repeat(n,1)
        weights=torch.tensor([[.5,.25,.25]]).repeat(n,1)
        result={"candidate_predictions":candidates,"routing_weights":weights,
                "mean":(candidates*weights).sum(1),"raw_anchor_mean":candidates[:,0],
                "sarn_endpoint_mean":candidates[:,1],"relation_available":sarn_active,
                "polar_direction_posterior":torch.full((n,36),1/36),
                "polar_concentration":torch.zeros(n),"polar_entropy":torch.ones(n)}
        features={"raw_representation":torch.zeros(n,1280),"sarn_representation":torch.ones(n,1280),
                  "geometry_features":torch.zeros(n,13)}
        return result,features


def test_cache_preserves_condition_major_ids_targets_and_groups(tmp_path):
    samples=[DirectSample('a','g1','g1',Path('a'),(0,0,4,4),.3),
             DirectSample('b','g2','g2',Path('b'),(0,0,4,4),.7)]
    batch={"sample_id":['a','b'],"target":torch.tensor([.3,.7]),
           "original_view":torch.zeros(2,3,4,4),"sarn_view":torch.zeros(2,3,4,4),
           "sarn_support_mask":torch.ones(2,1,4,4),"sarn_active":torch.ones(2,dtype=torch.bool),
           "raw_to_sarn_homography":torch.eye(3).repeat(2,1,1)}
    dest=tmp_path/'cache.pt'
    with patch('experiments.revision_cache._loader',return_value=[batch]):
        cache,_=collect_revision_cache(Reader(),samples,cache_path=dest,device=torch.device('cpu'),
            seed=20262020,batch_size=2,workers=0,scope='evaluation',availability_policy='image_only')
    payload=torch.load(dest,weights_only=False)
    assert payload['sample_ids']==['a','b']*6
    assert payload['group_ids']==['g1','g2']*6
    assert payload['conditions']==list(CONDITIONS)
    assert cache['condition_index'].tolist()==[i for i in range(6) for _ in range(2)]
    assert torch.equal(cache['target'],torch.tensor([.3,.7]*6))
    assert torch.allclose(cache['full_prediction'],torch.full((12,),.35))
    assert cache['moment_residual'].abs().max()==0
    assert bool(cache['relation_available'].all())


def test_availability_policy_is_explicit_and_reproducible():
    active=torch.tensor([True,False])
    assert effective_availability(active,'clean','image_only').tolist()==[True,False]
    assert effective_availability(active,'clean','clean_disabled').tolist()==[False,False]
    assert effective_availability(active,'perspective_severe','clean_disabled').tolist()==[True,False]


def test_syncg_uses_scene_groups_not_gauge_scene_combinations(tmp_path):
    import json
    from experiments.revision_cache import load_revision_samples
    image=tmp_path/'a.png';image.touch()
    manifest=tmp_path/'train.jsonl'
    manifest.write_text(json.dumps({'dataset':'SyncG','split':'train'})+'\n')
    samples=(DirectSample('a','type1::scene','scene',image,(0,0,4,4),.3),
             DirectSample('b','type2::scene','scene',image,(0,0,4,4),.7))
    with patch('experiments.revision_cache.load_syncg_samples',return_value=samples), \
         patch.dict('experiments.revision_cache.COHORT_COUNTS',{'syncg_train':(2,1)}):
        loaded=load_revision_samples(manifest,'syncg_train')
    assert [s.group_id for s in loaded]==['scene','scene']
