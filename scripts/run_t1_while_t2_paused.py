"""Evaluate the fixed T1 protocol during an explicitly authorized T2 pause."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

from priority_t1_delivery import LABELS, read, write, report


def run(root):
    frozen = root/'runtime/retrain-lr-v1-final'
    out = root/'runs/experiments/retrain-lr-v1'
    dest = out/'priority-T1'
    lock = (dest/'priority.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    suite_lock = (out/'suite.lock').open('a')
    fcntl.flock(suite_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(frozen/'scripts'))
    from neuroadapter_research.checkpoint import verify_checkpoint
    status = read(out/'T2/status.json')
    assert status['status'] == 'interrupted' and 0 < status['completed_updates'] < 159375
    point = out/'T2/checkpoints'/f'checkpoint-update-{status["completed_updates"]:08d}'
    verify_checkpoint(point, expected_world_size=2)
    trainer = read(point/'trainer_state.json')
    assert trainer['config_hash'] == status['config_hash'] and trainer['arm'] == 'T2'
    assert trainer['next_update'] == status['completed_updates']
    state = {'pid': os.getpid(), 'status': 'evaluating_paused_T2', 'stages': [],
             'T2_resume_checkpoint': str(point), 'T2_completed_updates': status['completed_updates'],
             'source_commit': os.environ['PRIORITY_SOURCE_COMMIT']}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1',
        PYTHONPATH=str(root/'runtime/subject01-4090-1a1fcfa/src'), CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    write(dest/'status.json', state)
    try:
        for label in LABELS:
            for phase in ('decode', 'score'):
                busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
                assert not busy, f'GPU occupied: {busy}'
                cmd = [sys.executable, str(frozen/'scripts/evaluate_retrain_lr.py'), '--root', str(root), '--phase', phase, '--label', label]
                item = {'label': label, 'phase': phase, 'command': cmd, 'exit_code': None,
                        'start': datetime.now(timezone.utc).isoformat()}
                state['stages'].append(item)
                state['status'] = phase+'-'+label
                with (dest/f'{phase}-{label}.log').open('a') as log:
                    child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                    item['pid'] = child.pid
                    write(dest/'status.json', state)
                    item['exit_code'] = child.wait()
                item['end'] = datetime.now(timezone.utc).isoformat()
                write(dest/'status.json', state)
                assert item['exit_code'] == 0, item
        report(root, frozen, out)
        state['status'] = 'T1_results_ready_visual_review_pending'
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        fcntl.flock(suite_lock, fcntl.LOCK_UN)
        cmd = ['env', 'PYTHONPATH='+str(root/'runtime/subject01-4090-1a1fcfa/src'),
               'CUDA_VISIBLE_DEVICES=0,1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
               'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'PYTHONUNBUFFERED=1',
               sys.executable, str(frozen/'scripts/run_retrain_lr_suite.py'), '--root', str(root)]
        shell = shlex.join(cmd)+' >> '+shlex.quote(str(dest/'resumed-controller.log'))+' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', 'neuroadapter-retrain-lr-resumed', shell], check=True)
        state['resume_controller_launched'] = True
        write(dest/'status.json', state)
        with (root/'repo/EXPERIMENT_LOG.md').open('a', encoding='utf-8') as log:
            log.write(f'\n\n### 暂停T2期间优先评价 {datetime.now(timezone.utc).isoformat()}\n\n'
                      f'状态{state["status"]}；已启动原冻结控制器，从完整恢复点{point}恢复T2。'
                      '实际恢复成功须核验T2新日志和进程；完整退出码见priority-T1/status.json。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
