"""One-time public-author/current data and FP32 training-chain audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from generator_semantic_core import base_config, output, source_identity, source_snapshot, spec
from neuroadapter_research.atomic import write_json_atomic
from neuroadapter_research.data import Subject1TrainingDataset
from neuroadapter_research.modeling import (NeuroAdapterTrainingModule, build_adapter,
    load_frozen_backbone, load_trainable_state_dict, trainable_state_dict)
from neuroadapter_research.trainer import min_snr_weights


class FixedPosterior:
    def __init__(self, original, sample): self.original,self.fixed=original,sample
    def sample(self): return self.fixed
    def __getattr__(self,name): return getattr(self.original,name)


class FixedEncode:
    def __init__(self, original, sample): self.latent_dist=FixedPosterior(original.latent_dist,sample)


class VAEProxy:
    def __init__(self, encode, sample): self.original_encode,self.sample=encode,sample
    def encode(self, images): return FixedEncode(self.original_encode(images),self.sample)


def tensor_differences(left, right):
    keys=set(left);assert keys==set(right);maximum=0.;rms=[]
    for key in keys:
        value=(left[key].float()-right[key].float()).abs();maximum=max(maximum,float(value.max()));rms.append(value.square().mean())
    return maximum,float(torch.stack(rms).mean().sqrt())


def main(root: Path) -> None:
    root=Path(root);cfg=base_config(root);protocol=spec();target=output(root)/"chain-audit";target.mkdir(parents=True,exist_ok=True)
    ids=np.loadtxt(cfg.paths["split_ids"],dtype=np.int64)
    fixed=sorted(ids.tolist(),key=lambda value:__import__('hashlib').sha256(f"generator-semantic-last-v1:{value}".encode()).digest())[:16]
    dataset=Subject1TrainingDataset(cfg.paths["training_cache"],cfg.paths["stimuli"],cfg.paths["split_ids"])
    lookup={int(value):index for index,value in enumerate(dataset.image_ids)}
    metadata=np.load(root/"data/derived/neural_data/metadata_sub-01.npy",allow_pickle=True).item()
    presentation=np.asarray(metadata["img_presentation_order"])
    parcel_dir=root/"data/derived/parcels/schaefer"
    parcels={hemi:torch.load(parcel_dir/f"{hemi}_labels_s01.pt",weights_only=True)[1:] for hemi in ("lh","rh")}
    selected={}
    for hemi in ("lh","rh"):
        scores=np.asarray([np.asarray(metadata[f"{hemi}_ncsnr"])[vertices.numpy()].mean() for vertices in parcels[hemi]])
        selected[hemi]=np.argsort(scores)[::-1][:100]
    data_checks=[]
    with h5py.File(root/"data/derived/neural_data/betas_sub-01.h5","r") as raw:
        for image_id in fixed:
            sample=dataset[lookup[image_id]];rows=np.flatnonzero(presentation==image_id);assert rows.shape==(3,)
            rebuilt=[]
            for hemi in ("lh","rh"):
                average=np.asarray(raw[f"{hemi}_betas"][rows],dtype=np.float32).mean(axis=0,dtype=np.float32)
                for parcel_index in selected[hemi]:
                    values=average[parcels[hemi][int(parcel_index)].numpy()]
                    padded=np.zeros(dataset.max_voxels,dtype=np.float32);padded[:len(values)]=values;rebuilt.append(padded)
            rebuilt=torch.from_numpy(np.stack(rebuilt));difference=(rebuilt-sample["brain"]).abs()
            data_checks.append({"image_id":image_id,"brain_max_abs":float(difference.max()),
                                "brain_exact":bool(torch.equal(rebuilt,sample["brain"])),
                                "image_shape":list(sample["image"].shape),"token_shape":list(sample["brain"].shape)})
    assert all(item["brain_exact"] for item in data_checks)
    device=torch.device("cuda:0");backbone=load_frozen_backbone(cfg.paths["stable_diffusion"])
    bundle=build_adapter(backbone.unet,200,626);state=torch.load(source_snapshot(root)/"model.pt",map_location="cpu",weights_only=True)
    load_trainable_state_dict(bundle,state);module=NeuroAdapterTrainingModule(bundle).to(device)
    backbone.vae.to(device).eval().requires_grad_(False);backbone.text_encoder.to(device).eval().requires_grad_(False)
    batch=[dataset[lookup[value]] for value in fixed[:4]]
    images=torch.stack([value["image"] for value in batch]).to(device);brain=torch.stack([value["brain"] for value in batch]).to(device)
    text_ids=backbone.tokenizer("",max_length=backbone.tokenizer.model_max_length,padding="max_length",truncation=True,return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        posterior=backbone.vae.encode(images).latent_dist;generator=torch.Generator(device=device).manual_seed(20260918)
        epsilon=torch.randn(posterior.mean.shape,device=device,generator=generator);latent_sample=posterior.mean+posterior.std*epsilon
        latent=latent_sample*backbone.vae.config.scaling_factor;noise=torch.randn(latent.shape,device=device,generator=generator)
        timesteps=torch.tensor([50,200,500,800],device=device);text=backbone.text_encoder(text_ids)[0].expand(4,-1,-1)
        noisy=backbone.noise_scheduler.add_noise(latent,noise,timesteps);keep=(torch.rand((4,200,768),device=device,generator=generator)>0.10).float()
    optimizer=torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-6,foreach=False,fused=False)
    # The frozen construction primitives import their own vendor tree first.  The
    # author-function audit must instead import train_brain_adapter.py and all of
    # its brain_adapter dependencies from one matching public source tree.
    for name in [name for name in sys.modules if name == "brain_adapter" or name.startswith("brain_adapter.")]:
        del sys.modules[name]
    sys.path.insert(0,str(root/"repo/vendor/NeuroAdapter"))
    # The released train file imports an optional dataset class that is absent
    # from the pinned scientific dataset.py.  It is not used by
    # process_training_batch; provide an import-only placeholder and record this
    # public-source compatibility patch in the audit output.
    import brain_adapter.dataset as author_dataset
    if not hasattr(author_dataset,"nsd_groupwise_topk_parcel_dataset"):
        author_dataset.nsd_groupwise_topk_parcel_dataset=author_dataset.nsd_topk_parcel_dataset
    import train_brain_adapter as author
    author_batch={"img_ipadapter":images,"text_input_ids":text_ids.expand(4,-1),"brain_lh_f":brain[:,:100],"brain_rh_f":brain[:,100:]}
    accelerator=SimpleNamespace(device=device)
    process_cpu_rng=torch.get_rng_state().clone();process_cuda_rng=torch.cuda.get_rng_state(device).clone()
    original_encode=backbone.vae.encode
    with patch.object(backbone.vae,"encode",VAEProxy(original_encode,latent_sample).encode), \
         patch.object(torch,"randn_like",lambda value:noise), \
         patch.object(torch,"randint",lambda *args,**kwargs:timesteps), \
         patch.object(author,"apply_fmri_token_dropout",lambda value: value*keep):
        author_loss=author.process_training_batch(author_batch,backbone.vae,backbone.noise_scheduler,
            backbone.text_encoder,bundle.guidance_generator,bundle.neuro_adapter,torch.float32,accelerator,
            SimpleNamespace(num_parcels=200,max_voxels=626))
    author_loss.backward();author_grads={name:p.grad.detach().cpu().clone() for name,p in module.named_parameters() if p.requires_grad};optimizer.step()
    author_updated={group:{name:value.clone() for name,value in values.items()} for group,values in trainable_state_dict(bundle).items()}
    load_trainable_state_dict(bundle,state);optimizer=torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-6,foreach=False,fused=False);optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(process_cpu_rng);torch.cuda.set_rng_state(process_cuda_rng,device)
    prediction=module(noisy,timesteps,text,brain,keep)
    per=F.mse_loss(prediction.float(),noise.float(),reduction="none").mean((1,2,3));current_loss=(per*min_snr_weights(timesteps,backbone.noise_scheduler.alphas_cumprod.to(device),5.)).mean()
    current_loss.backward();current_grads={name:p.grad.detach().cpu().clone() for name,p in module.named_parameters() if p.requires_grad};optimizer.step()
    current_updated=trainable_state_dict(bundle)
    gradient_diff=tensor_differences(author_grads,current_grads)
    state_diffs={group:tensor_differences(author_updated[group],current_updated[group]) for group in author_updated}
    dtype_record={"vae_mean":str(posterior.mean.dtype),"vae_std":str(posterior.std.dtype),"vae_epsilon":str(epsilon.dtype),
        "vae_sample":str(latent_sample.dtype),"scaled_latent":str(latent.dtype),"diffusion_noise":str(noise.dtype),
        "alphas":str(backbone.noise_scheduler.alphas_cumprod.dtype),"noisy_latent":str(noisy.dtype),
        "unet_input":str(noisy.dtype),"unet_output":str(prediction.dtype),"scheduler_config":dict(backbone.noise_scheduler.config)}
    # Adam's first update can differ by about one LR on isolated near-zero
    # coordinates when sub-micro floating-point noise flips a gradient sign.
    # Use the full-state RMS together with loss/gradient bounds, while retaining
    # every maximum difference in the report instead of hiding it.
    passed=(abs(float(author_loss)-float(current_loss))<1e-6 and
            gradient_diff[0]<1e-5 and gradient_diff[1]<1e-6 and
            all(value[1]<1e-6 for value in state_diffs.values()))
    write_json_atomic(target/"audit.json",{"status":"passed" if passed else "blocked","fixed_image_ids":fixed,
        "data_checks":data_checks,"author_loss":float(author_loss),"current_loss":float(current_loss),
        "gradient_max_abs":gradient_diff[0],"gradient_rms":gradient_diff[1],"updated_state_differences":state_diffs,
        "dtypes":dtype_record,"random_injection":"explicit VAE epsilon-derived sample, diffusion noise, timesteps and token mask",
        "author_function":"vendor/NeuroAdapter/train_brain_adapter.py::process_training_batch",
        "author_import_patch":"unused nsd_groupwise_topk_parcel_dataset alias required by released top-level import",
        "source":source_identity(root),"scope":"public-code correspondence only; not unreleased author HDF5/config proof"})
    if not passed: raise RuntimeError("public author/current chain audit found a blocking difference")
    dataset.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);main(parser.parse_args().root)
