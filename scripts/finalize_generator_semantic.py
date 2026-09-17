"""Export the bounded experiment, fixed gallery, and machine-readable decision."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
from PIL import Image, ImageDraw

from generator_semantic_core import output, source_snapshot, spec
from neuroadapter_research.atomic import sha256_file, write_json_atomic


def image(path: Path, size: int = 160) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def make_galleries(root: Path, final: Path) -> list[dict]:
    out=output(root);protocol=json.loads((out/"evaluation/protocol.json").read_text())
    previous=json.loads((root/"runs/experiments/retrain-lr-v1/evaluation/protocol.json").read_text())
    b0=Path(previous["references"]["OLD-239063"]["folder"])
    folders={"B0":b0,"K":out/"evaluation/K-5000","M":out/"evaluation/M-5000"}
    manifests={name:json.loads((folder/"decode_manifest.json").read_text()) for name,folder in folders.items()}
    lookup={name:{int(record["image_id"]):record for record in manifest["records"]} for name,manifest in manifests.items()}
    gallery=final/"galleries";gallery.mkdir(parents=True,exist_ok=True)
    records=[];size=160;headers=["GT","B0 seed0","B0 seed1","K seed0","K seed1","M seed0","M seed1"]
    with h5py.File(root/"data/raw/nsd/stimuli/nsd_stimuli.hdf5","r") as h5:
        for page in range(4):
            selected=protocol["visual_ids"][page*8:(page+1)*8]
            canvas=Image.new("RGB",(len(headers)*size,(len(selected)+1)*size),"white");draw=ImageDraw.Draw(canvas)
            for column,label in enumerate(headers): draw.text((column*size+5,5),label,fill="black")
            for row,image_id in enumerate(selected,1):
                gt=Image.fromarray(h5["imgBrick"][image_id]).convert("RGB").resize((size,size),Image.Resampling.LANCZOS)
                values=[gt]
                for name in ("B0","K","M"):
                    record=lookup[name][image_id]
                    values.extend([image(folders[name]/item["path"],size) for item in record["files"]])
                for column,value in enumerate(values): canvas.paste(value,(column*size,row*size))
                draw.text((5,row*size+5),str(image_id),fill="white",stroke_width=2,stroke_fill="black")
                records.append({"page":page+1,"image_id":image_id})
            path=gallery/f"fixed32-page-{page+1}.jpg";canvas.save(path,quality=94)
    return records


def main(root: Path) -> None:
    out=output(root);evaluation=out/"evaluation";gate=json.loads((evaluation/"pilot-gate.json").read_text())
    assert not gate["gate"]["passed_numeric"]
    rows={row["label"]:row for row in gate["rows"]};new=[rows["K-5000"],rows["M-5000"]]
    developed=max(new,key=lambda row:(row["semantic_score"],row["PixCorr"],-int(row["label"].split("-")[1])))
    assert developed["label"]=="K-5000"
    final=out/"final";candidate=final/"developed_candidate";candidate.mkdir(parents=True,exist_ok=True)
    source=out/"K/snapshots/snapshot-update-00005000/model.pt";target=candidate/"generator_state.pt"
    shutil.copy2(source,target)
    snapshot_metadata=json.loads((out/"K/snapshots/snapshot-update-00005000/metadata.json").read_text())
    metadata={"experiment":spec()["experiment_type"],"role":"developed_candidate_not_recommended_replacement",
        "label":"K-5000","local_update":5000,"source_update":spec()["source_update"],
        "source_R_sha256":spec()["source_sha256"],"generator_state_sha256":sha256_file(target),
        "training_implementation_commit":snapshot_metadata["implementation_commit"],
        "training_config_hash":snapshot_metadata["config_hash"],
        "training_objective":"original diffusion loss only control","inference_requires_training_CLIP":False,
        "semantic_score":developed["semantic_score"],"pixcorr":developed["PixCorr"]}
    write_json_atomic(candidate/"metadata.json",metadata)
    (candidate/"LOAD_EXAMPLE.py").write_text(
        "state = torch.load('generator_state.pt', map_location='cpu', weights_only=True)\n"
        "load_trainable_state_dict(bundle, state)\n",encoding="utf-8")
    write_json_atomic(final/"recommended_generator.json",{
        "recommended_generator":"B0","reason":"best new candidate failed the frozen 1 pp replacement threshold",
        "source":str(source_snapshot(root)/"model.pt"),"sha256":spec()["source_sha256"]})
    records=make_galleries(root,final)
    write_json_atomic(final/"decision.json",{"status":"complete_no_extension","pilot_gate":gate["gate"],
        "developed_candidate":metadata,"recommended_generator":"B0","completed_snapshots":["K-5000","M-5000"],
        "standard_test_used":False,"additional_training_authorized":False,"gallery_records":records,
        "active_training_seconds":json.loads((out/"K/status.json").read_text())["seconds"]+json.loads((out/"M/status.json").read_text())["seconds"]})
    report="# generator-semantic-last-v1 最终结果\n\n"
    report+=f"本轮从旧 R（239063，`{spec()['source_sha256']}`）分别运行 K5000 和 M5000。冻结数值门控未通过，因此 M 未延长到 20000，也未启动额外实验。\n\n"
    report+="## 同协议结果\n\n|权重|S|N|PixCorr|SSIM|AlexNet2|AlexNet5|Inception|CLIP ID|EfficientNet|SwAV|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for label in ("B0","K-5000","M-5000"):
        r=rows[label];report+=f"|{label}|{r['semantic_score']:.4f}|{r['non_clip_high_level']:.4f}|{r['PixCorr']:.6f}|{r['SSIM']:.6f}|{r['AlexNet2']:.4f}|{r['AlexNet5']:.4f}|{r['Inception']:.4f}|{r['CLIP_identification']:.4f}|{r['EfficientNet']:.6f}|{r['SwAV']:.6f}|\n"
    d=gate["gate"]["deltas"];report+=f"\nM5000 相对 B0 的 S 差值为 {d['S_vs_B0']:.4f} 个百分点，相对 K5000 为 {d['S_vs_K']:.4f}；PixCorr 相对 B0 为 {d['PixCorr_vs_B0']:.6f}。数值门控四项中仅 PixCorr 下限通过。\n\n"
    report+="## 权重决定\n\n本轮 developed_candidate 为 K5000，但其 S 相对 B0 仅提高 {:.4f} 个百分点，低于预定的 1 个百分点替换门槛。因此 recommended_generator 继续为 B0，K5000 仍按要求完整导出。\n\n".format(rows['K-5000']['semantic_score']-rows['B0']['semantic_score'])
    report+="## 固定图册\n\n"+"\n".join(f"![第{i}页](galleries/fixed32-page-{i}.jpg)" for i in range(1,5))+"\n\n## 视觉检查\n\n待实际打开四页图册后补充主体、动作、场景、布局、复制和伪影记录。\n"
    (final/"REPORT_ZH.md").write_text(report,encoding="utf-8")
    print(json.dumps({"developed_candidate":metadata,"recommended_generator":"B0"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);main(parser.parse_args().root)
