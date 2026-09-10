"""Publish the selected snapshot pointer, never datasets or image previews."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    repo_id = 'gugabobo/neuroadapter-subject1-selection-4090'
    token = (root / 'credentials/hf-upload/token').read_text().strip()
    api = HfApi(token=token)
    assert api.whoami()['name'] == 'gugabobo'
    info = api.model_info(repo_id, files_metadata=True)
    assert not info.private
    evidence = root / 'repo/manifests/selection-20260910'
    lock = json.loads((evidence / 'RESEARCH_WEIGHT_LOCK.json').read_text())
    assert lock['no_more_training'] and lock['selected_update'] == 239063
    model_path = 'snapshots/snapshot-update-00239063/model.pt'
    item = next(item for item in info.siblings if item.rfilename == model_path)
    assert item.lfs.sha256 == lock['model_sha256']
    cache = root / 'cache/hf-upload-verification'
    readme = Path(hf_hub_download(repo_id, 'README.md', revision=info.sha, token=token, cache_dir=cache)).read_text()
    marker = '## Selected research weight / 选定研究权重'
    assert marker not in readme, 'preserve an existing published selection; inspect before retrying'
    readme += '\n' + marker + '\n\n2026-09-10 已完成选优，固定使用 **239063 步**，不再续训或重训。\n\n下载 [snapshot-update-00239063](snapshots/snapshot-update-00239063) 目录全部文件。\n\n模型 SHA-256：`' + lock['model_sha256'] + '`。\n\n真实训练图数 8500，内部验证 500；未进行 9000 图重训，标准测试未用于选择。选定模型并不代表达到论文性能。\n\n[中文选优报告](selection/SELECTION_REPORT.md) · [权重锁定记录](selection/RESEARCH_WEIGHT_LOCK.json) · [完整指标汇总](selection/metrics_summary.json) · [复核记录](selection/completion_audit.json)。\n'
    payloads = {'README.md': readme.encode('utf-8'), 'selection/SELECTION_REPORT.md': (root / 'repo/docs/SELECTION_RESULT.md').read_bytes()}
    for p in sorted(evidence.glob('*.json')):
        payloads['selection/' + p.name] = p.read_bytes()
    assert all(name == 'README.md' or name.startswith('selection/') and name.endswith(('.json', '.md')) for name in payloads)
    commit = api.create_commit(repo_id, operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in payloads.items()], commit_message='Publish locked Subject 1 research weight and validation selection results', parent_commit=info.sha)
    for name, data in payloads.items():
        downloaded = Path(hf_hub_download(repo_id, name, revision=commit.oid, token=token, cache_dir=cache))
        assert hashlib.sha256(downloaded.read_bytes()).digest() == hashlib.sha256(data).digest()
    final = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
    assert not final.private
    item = next(item for item in final.siblings if item.rfilename == model_path)
    assert item.lfs.sha256 == lock['model_sha256']
    result = {'status': 'verified', 'repo_id': repo_id, 'public': True, 'revision': commit.oid, 'selected_update': lock['selected_update'], 'model_path': model_path, 'model_sha256': item.lfs.sha256, 'published_file_count': len(payloads), 'no_more_training': True}
    (root / 'artifacts/hf-selection-published-20260910.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
