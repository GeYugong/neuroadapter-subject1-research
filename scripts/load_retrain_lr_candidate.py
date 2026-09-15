"""Load the exported generator alone and reconstruct one fixed validation image."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
import diagnose_condition_path as diag
from retrain_lr_core import settings, output
from neuroadapter_research.atomic import sha256_file
from neuroadapter_research.backend import configure_torch_backend
from neuroadapter_research.data import Subject1TrainingDataset


@torch.no_grad()
def run(root,destination):
    assert destination.resolve().is_relative_to(root.resolve())
    destination.mkdir(parents=True,exist_ok=False)
    base,_,_=settings(root,'T1');configure_torch_backend(base.training)
    out=output(root);model=out/'best_new_candidate/model.pt'
    meta=json.loads((model.parent/'metadata.json').read_text());assert sha256_file(model)==meta['model_sha256']
    protocol=json.loads((out/'evaluation/protocol.json').read_text())
    ds=Subject1TrainingDataset(base.paths['training_cache'],base.paths['stimuli'],base.paths['validation_ids'])
    backbone,bundle=diag.load_models(base,model,torch.device('cuda:0'),torch.bfloat16)
    i=protocol['ids'][0]
    with patch.object(diag,'NAMESPACE',protocol['namespace']):
        images=diag.generate(backbone,bundle,ds[0]['brain'],i,'validation',torch.device('cuda:0'),torch.bfloat16,4.,
            torch.load(root/protocol['noise_paths'][str(i)],weights_only=True))
    for c,image in enumerate(images):diag.png(destination/f'{i}-candidate-{c}.png',image)
    ds.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();run(a.root,a.destination)
