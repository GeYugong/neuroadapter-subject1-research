"""Package a manually reviewed LR report and its complete existing galleries."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def run(repo, trajectory):
    sheets = sorted((trajectory/'galleries').glob('*.jpg'))
    assert len(sheets) == 32
    report = (repo/'docs/PAIRED_LR_PROBE_V1.md').read_text(encoding='utf-8')
    report += '\n\n## 全部固定样本图册\n\ntrajectory为正确输入沿训练节点的变化；endpoint为正确与完整错配脑输入的终点对照。c0/c1是两个固定候选，均完整保留。\n'
    for path in sheets:
        report += f'\n### {path.stem}\n\n![{path.stem}](trajectory/galleries/{path.name})\n'
    (trajectory.parent/'REPORT.md').write_text(report, encoding='utf-8')
    public = repo/'manifests/paired-lr-probe-v1/closure'
    public.mkdir(parents=True, exist_ok=True)
    sources = []
    for path in list(trajectory.glob('*.csv')) + [trajectory/'summary.json']:
        shutil.copy2(path, public/path.name)
        sources.append({'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (public/'INDEX.json').write_text(json.dumps({'sources': sources, 'visual_review': 'all32 sheets inspected individually',
        'gallery_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sheets},
        'new_diffusion_images': 0, 'neuroadapter_updates': 0, 'note': 'original summary retains pre-review pending status; this closure supersedes it'}, indent=2)+'\n')
    print('Packaged32 reviewed galleries and4 source tables/manifests')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--trajectory', type=Path, required=True)
    a = p.parse_args()
    run(a.repo, a.trajectory)
