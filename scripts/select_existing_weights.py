#!/usr/bin/env python3
"""Evaluate frozen snapshots on two GPUs, select once, and stop without training."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from neuroadapter_research.atomic import sha256_file, write_json_atomic
from neuroadapter_research.config import load_training_config
from neuroadapter_research.protocol import (
    load_selection_plan, method_fingerprint, validate_selection_snapshot,
    validate_selection_plan_inputs, verify_protocol_repository,
)


ALLOWED_SCRIPTS = {'validation_loss.py', 'decode_validation.py', 'evaluate_validation.py', 'select_checkpoint.py'}


def log_entry(root, text):
    with (root / 'repo/EXPERIMENT_LOG.md').open('a', encoding='utf-8') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f'\n### {datetime.now().astimezone().isoformat()}：现有权重选优\n\n{text}\n')
        handle.flush()


def command(runtime, name, args):
    if name not in ALLOWED_SCRIPTS:
        raise ValueError('selection controller is forbidden to launch training or arbitrary scripts')
    return [sys.executable, str(runtime / 'scripts' / name), *map(str, args)]


def run_step(root, runtime, work, name, arguments, gpu=None):
    work.mkdir(parents=True, exist_ok=True)
    output = work / f'{name}.log'
    status = work / f'{name}.status.json'
    if output.exists() or status.exists():
        raise FileExistsError(f'preserving existing attempt: {output}')
    env = os.environ.copy()
    env.update(PYTHONPATH=str(runtime / 'src'), PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    argv = command(runtime, name, arguments)
    log_entry(root, f'阶段目录：`{work.relative_to(root)}`；GPU：`{gpu}`；控制台：`{output.relative_to(root)}`。\n\n```bash\nCUDA_VISIBLE_DEVICES={gpu if gpu is not None else ""} PYTHONPATH={runtime}/src {shlex.join(argv)}\n```')
    started = time.monotonic()
    write_json_atomic(status, {'status': 'running', 'gpu': gpu, 'command': argv, 'started_at': datetime.now().astimezone().isoformat()})
    with output.open('x') as handle:
        result = subprocess.run(argv, env=env, stdout=handle, stderr=subprocess.STDOUT)
    write_json_atomic(status, {'status': 'completed' if result.returncode == 0 else 'failed', 'exit_code': result.returncode, 'gpu': gpu, 'command': argv, 'elapsed_seconds': time.monotonic() - started})
    log_entry(root, f'`{work.relative_to(root)}/{name}` 退出码 {result.returncode}，耗时 {time.monotonic() - started:.1f} 秒。')
    if result.returncode:
        raise RuntimeError(f'{name} failed; see {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load_training_config(args.config, require_frozen=True)
    root = Path(config.raw['project_root']).resolve()
    runtime = args.runtime.resolve()
    out = args.output.resolve()
    out.relative_to(root)
    out.mkdir(parents=True, exist_ok=False)
    progress = out / 'status.json'
    write_json_atomic(progress, {'status': 'validating', 'no_more_training': True})
    try:
        verify_protocol_repository(runtime, config.raw['protocol_commit'])
        plan = load_selection_plan(config.paths['selection_plan'], require_frozen=True)
        validate_selection_plan_inputs(plan, validation_ids_path=config.paths['validation_ids'], repository_root=runtime)
        fingerprint = method_fingerprint(config)
        run = config.paths['output_dir']
        training_status = json.loads((run / 'run_status.json').read_text())
        if training_status.get('status') != 'completed' or training_status.get('run_kind') != 'selection':
            raise ValueError('only completed formal selection runs are eligible')
        snapshots = {}
        for update in plan.raw['expected_snapshot_updates']:
            path = run / 'snapshots' / f'snapshot-update-{update:08d}'
            validate_selection_snapshot(path, config=config, plan=plan, fingerprint=fingerprint)
            snapshots[update] = path
        subprocess.run(['bash', str(runtime / 'scripts/check_gpu_idle.sh')], check=True)
        write_json_atomic(out / 'policy.json', {
            'decision_date': '2026-09-10', 'no_more_training': True,
            'training_images': 8500, 'validation_images': 500,
            'standard_test_used_for_selection': False,
            'runtime_commit': config.raw['protocol_commit'], 'config_sha256': config.sha256,
            'controller_sha256': sha256_file(Path(__file__)),
            'expected_updates': list(snapshots), 'method_fingerprint': fingerprint,
        })

        def checkpoint_task(update, stage, gpu):
            work = out / stage / f'update-{update:08d}'
            work.mkdir(parents=True, exist_ok=False)
            loss_dir = work if stage == 'screening' else out / 'screening' / work.name
            if stage == 'screening':
                run_step(root, runtime, work, 'validation_loss.py', ['--config', args.config, '--snapshot', snapshots[update], '--output-json', work / 'loss.json', '--output-csv', work / 'loss.csv'], gpu)
            run_step(root, runtime, work, 'decode_validation.py', ['--config', args.config, '--snapshot', snapshots[update], '--stage', stage, '--output-dir', work / 'decode'], gpu)
            run_step(root, runtime, work, 'evaluate_validation.py', ['--config', args.config, '--decode-manifest', work / 'decode/decode_manifest.json', '--validation-loss', loss_dir / 'loss.json', '--output-json', work / 'evaluation.json', '--output-csv', work / 'evaluation.csv'], gpu)
            return work / 'evaluation.json'

        def phase(stage, updates):
            write_json_atomic(progress, {'status': 'running', 'stage': stage, 'updates': updates, 'no_more_training': True})
            def worker(gpu):
                return [checkpoint_task(update, stage, gpu) for update in updates[gpu::2]]
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker, gpu) for gpu in (0, 1)]
                return sorted(path for future in futures for path in future.result())

        screening = phase('screening', list(snapshots))
        shortlist_path = out / 'shortlist.json'
        run_step(root, runtime, out / 'shortlist', 'select_checkpoint.py', ['--config', args.config, '--stage', 'shortlist', '--input', *screening, '--output', shortlist_path])
        shortlisted = json.loads(shortlist_path.read_text())['shortlist_updates']
        final_results = phase('final', shortlisted)
        final_path = out / 'final_selection.json'
        run_step(root, runtime, out / 'selection', 'select_checkpoint.py', ['--config', args.config, '--stage', 'final', '--shortlist-manifest', shortlist_path, '--input', *final_results, '--output', final_path])
        result = json.loads(final_path.read_text())
        chosen = int(result['selected_update_u_star'])
        provenance = validate_selection_snapshot(snapshots[chosen], config=config, plan=plan, fingerprint=fingerprint)
        destination = out / 'selected_snapshot'
        destination.mkdir()
        for path in snapshots[chosen].iterdir():
            os.link(path, destination / path.name)
        lock = {
            'schema_version': 1, 'status': 'selected_existing_weight', 'no_more_training': True,
            'training_run_kind': 'selection', 'training_image_count': 8500,
            'validation_image_count': 500, 'retrained_on_9000': False,
            'standard_test_used': False, 'selected_update': chosen,
            'model_sha256': provenance['model_sha256'],
            'snapshot_relative_path': str(snapshots[chosen].relative_to(root)),
            'selected_snapshot_relative_path': str(destination.relative_to(root)),
            'selection_manifest_sha256': sha256_file(final_path),
            'protocol_commit': config.raw['protocol_commit'], 'config_sha256': config.sha256,
            'method_fingerprint': fingerprint,
        }
        write_json_atomic(out / 'RESEARCH_WEIGHT_LOCK.json', lock)
        metrics = json.loads((out / 'final' / f'update-{chosen:08d}' / 'evaluation.json').read_text())['checkpoints'][0]['metrics']
        report = f'# 现有权重选优结果\n\n选定更新步数：**{chosen}**。\n\n模型 SHA-256：`{provenance["model_sha256"]}`。\n\n该权重来自 8500 张训练图的已完成 selection 运行，使用固定 500 张内部验证图及原定两阶段指标规则选择。未进行 9000 图重训，未使用标准测试集挑选。选定后停止，不再训练。\n\n## 验证指标\n\n| 指标 | 数值 |\n| --- | ---: |\n'
        report += ''.join(f'| {name} | {value:.8f} |\n' for name, value in metrics.items())
        report += '\n该结果是预定候选和验证协议下的选择，不保证在所有数据或后续任务上全局最优。图片位于对应 final/update 目录的 decode 子目录。\n'
        (out / 'SELECTION_REPORT.md').write_text(report, encoding='utf-8')
        write_json_atomic(progress, {'status': 'completed', 'selected_update': chosen, 'no_more_training': True, 'model_sha256': provenance['model_sha256'], 'completed_at': datetime.now().astimezone().isoformat()})
        log_entry(root, f'现有权重选优完成：update={chosen}，SHA=`{provenance["model_sha256"]}`；报告 `{out.relative_to(root)}/SELECTION_REPORT.md`。流程已停止，不执行续训或全量重训。')
    except BaseException as error:
        write_json_atomic(progress, {'status': 'failed', 'error': str(error), 'no_more_training': True})
        log_entry(root, f'选优流程异常停止：{error}。保留全部已有输出，不启动训练。')
        raise


if __name__ == '__main__':
    main()
