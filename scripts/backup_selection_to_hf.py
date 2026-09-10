#!/usr/bin/env python3
"""Publish only allowlisted completed selection artifacts using project credentials."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.checkpoint import verify_inference_snapshot
from neuroadapter_research.config import load_training_config
from neuroadapter_research.protocol import load_selection_plan


def sanitize(value, root):
    if isinstance(value, dict):
        return {key: sanitize(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), '${PROJECT_ROOT}')
    return value


def link_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise ValueError(f'backup staging collision: {target.name}')
        return
    os.link(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not args.repo_id.startswith('gugabobo/'):
        raise ValueError('only the explicitly authorized gugabobo account is allowed')
    token_path = root / 'credentials/hf-upload/token'
    token = token_path.read_text().strip()
    api = HfApi(endpoint='https://huggingface.co', token=token)
    identity = api.whoami()
    if identity.get('name') != 'gugabobo':
        raise ValueError('project credential account is not gugabobo')
    print('verified_account=gugabobo', flush=True)
    config = load_training_config(args.config, require_frozen=True)
    run = config.paths['output_dir']
    status = json.loads((run / 'run_status.json').read_text())
    if status.get('status') != 'completed' or status.get('run_kind') != 'selection':
        raise ValueError('backup requires a completed selection run')
    staging = root / 'artifacts/hf-public-selection-20260910'
    staging.mkdir(parents=True, exist_ok=True)
    plan = load_selection_plan(config.paths['selection_plan'], require_frozen=True)
    for update in plan.raw['expected_snapshot_updates']:
        source = run / 'snapshots' / f'snapshot-update-{update:08d}'
        verify_inference_snapshot(source)
        for name in ('model.pt', 'metadata.json', 'MANIFEST.json', 'COMPLETE'):
            link_file(source / name, staging / 'snapshots' / source.name / name)
    checkpoint = run / 'checkpoints' / f'checkpoint-update-{status["last_saved_update"]:08d}'
    manifest = json.loads((checkpoint / 'MANIFEST.json').read_text())
    complete = (checkpoint / 'COMPLETE').read_text().split()
    if complete != [sha256_file(checkpoint / 'MANIFEST.json'), 'MANIFEST.json']:
        raise ValueError('checkpoint completion manifest hash mismatch')
    allowed = {'model.pt', 'optimizer.pt', 'rank-00000.pt', 'rank-00001.pt', 'trainer_state.json'}
    if set(manifest['files']) != allowed:
        raise ValueError('unexpected checkpoint file inventory')
    for name in allowed:
        record = manifest['files'][name]
        if sha256_file(checkpoint / name) != record['sha256'] or (checkpoint / name).stat().st_size != record['size']:
            raise ValueError(f'checkpoint hash mismatch: {name}')
    for name in sorted(allowed | {'MANIFEST.json', 'COMPLETE'}):
        link_file(checkpoint / name, staging / 'resume-backup' / checkpoint.name / name)
    for name in ('effective_run.json', 'run_status.json'):
        write_json_atomic(staging / name, sanitize(json.loads((run / name).read_text()), root))
    link_file(run / 'training.jsonl', staging / 'training.jsonl')
    write_json_atomic(staging / 'training_config.json', sanitize(config.raw, root))
    write_json_atomic(staging / 'selection_plan.json', plan.raw)
    lock_text = config.paths['environment_lock'].read_text().replace(str(root), '${PROJECT_ROOT}')
    (staging / 'requirements-freeze.txt').write_text(lock_text)
    (staging / 'README.md').write_text('''---
language:
- zh
tags:
- neuroadapter
- fmri
- research
---
# NeuroAdapter Subject 1 候选权重备份

本仓库公开备份 GeYugong 的独立复现实验，不是论文作者发布的官方权重。

20 个 snapshot 来自 8500 张训练图的正式 selection 训练，另留 500 张图作内部验证。
目前候选权重备份完成不等于已经选出最佳权重，也不代表达到论文指标。
按 2026-09-10 决定，后续只从现有候选中选择研究用权重，不进行 9000 图全量重训。

`snapshots/` 可用于推理；`resume-backup/` 仅为完整状态归档，不授权继续训练。
本仓库不包含 NSD fMRI 数据、刺激图片、登录凭据或完整 Stable Diffusion 基础权重。
权重采用 PyTorch 格式，仅应加载可信来源，并按对应代码使用安全加载选项。

代码：https://github.com/GeYugong/neuroadapter-subject1-research
冻结训练与评价提交：1a1fcfa66e06de07a04dfbb48cc6f9ad108ed567
上游 NeuroAdapter：https://github.com/kriegeskorte-lab/NeuroAdapter
基础模型：Stable Diffusion v1.5，使用时须遵守相关上游模型和代码许可。
文件大小和 SHA-256 见 `BACKUP_MANIFEST.json`。
''', encoding='utf-8')
    files = [path for path in staging.rglob('*') if path.is_file() and '.cache' not in path.parts and path.name != 'BACKUP_MANIFEST.json']
    records = [{'path': path.relative_to(staging).as_posix(), 'size': path.stat().st_size, 'sha256': sha256_file(path)} for path in sorted(files)]
    write_json_atomic(staging / 'BACKUP_MANIFEST.json', {'schema_version': 1, 'repo_id': args.repo_id, 'source_config_sha256': config.sha256, 'files': records})
    try:
        existing = api.model_info(args.repo_id)
    except RepositoryNotFoundError:
        existing = None
    if existing is None:
        api.create_repo(args.repo_id, repo_type='model', private=False)
        api.upload_file(repo_id=args.repo_id, path_or_fileobj=staging / 'BACKUP_MANIFEST.json', path_in_repo='BACKUP_MANIFEST.json', commit_message='Initialize allowlisted selection backup inventory')
    else:
        prior = hf_hub_download(args.repo_id, 'BACKUP_MANIFEST.json', token=token, cache_dir=root / 'cache/hf-upload-verification')
        if json.loads(Path(prior).read_text()).get('source_config_sha256') != config.sha256:
            raise ValueError('existing HF repository is not this backup; refusing overwrite')
        if existing.private:
            raise ValueError('existing repository is private; refusing implicit visibility change')
    paths = [record['path'] for record in records] + ['BACKUP_MANIFEST.json']
    print(f'uploading_repo={args.repo_id} files={len(paths)} bytes={sum(r["size"] for r in records)}', flush=True)
    commit = api.upload_folder(repo_id=args.repo_id, folder_path=staging, allow_patterns=paths, commit_message='Back up completed Subject 1 selection weights')
    info = api.model_info(args.repo_id, revision=commit.oid, files_metadata=True)
    if info.private:
        raise ValueError('HF backup is not public')
    remote = {item.rfilename: item for item in info.siblings}
    for record in records + [{'path': 'BACKUP_MANIFEST.json', 'size': (staging / 'BACKUP_MANIFEST.json').stat().st_size, 'sha256': sha256_file(staging / 'BACKUP_MANIFEST.json')}]:
        item = remote[record['path']]
        if item.size != record['size']:
            raise ValueError(f'remote size mismatch: {record["path"]}')
        if item.lfs:
            digest = item.lfs.sha256 if hasattr(item.lfs, 'sha256') else item.lfs['sha256']
        else:
            downloaded = hf_hub_download(args.repo_id, record['path'], revision=commit.oid, token=token, cache_dir=root / 'cache/hf-upload-verification')
            digest = sha256_file(Path(downloaded))
        if digest != record['sha256']:
            raise ValueError(f'remote SHA mismatch: {record["path"]}')
    report = {'status': 'verified', 'repo_id': args.repo_id, 'url': f'https://huggingface.co/{args.repo_id}', 'revision': commit.oid, 'public': True, 'file_count': len(paths), 'snapshot_count': 20, 'source_config_sha256': config.sha256, 'backup_manifest_sha256': sha256_file(staging / 'BACKUP_MANIFEST.json')}
    write_json_atomic(root / 'artifacts/hf-backup-20260910.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
