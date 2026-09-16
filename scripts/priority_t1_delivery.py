"""One-off evaluation priority; never signal training workers or change training."""
import argparse
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

LABELS = ('T1-159375', 'OLD-159375', 'OLD-239063')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def proc(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return fields


def report(root, frozen, out):
    sys.path.insert(0, str(frozen/'scripts'))
    import numpy as np
    from PIL import Image, ImageDraw
    import h5py
    from retrain_lr_core import settings
    from neuroadapter_research.atomic import sha256_file
    ev = out/'evaluation'
    dest = out/'priority-T1'
    gallery = dest/'gallery'
    gallery.mkdir(exist_ok=True)
    protocol = read(ev/'protocol.json')
    summaries = {label: read(ev/f'{label}-summary.json') for label in LABELS}
    assert all(s['image_count'] == 500 and not s['smoke_only'] for s in summaries.values())
    metrics = list(summaries[LABELS[0]]['mean'])
    assert len(metrics) == 8
    semantic = lambda m: sum(m[k] for k in ('AlexNet5', 'Inception', 'CLIP_identification'))/3
    means = {label: {**s['mean'], 'semantic_score': semantic(s['mean'])} for label, s in summaries.items()}
    delta = {label: {k: means[LABELS[0]][k]-means[label][k] for k in means[label]} for label in LABELS[1:]}
    source = out/'T1/snapshots/snapshot-update-00159375'
    metadata = read(source/'metadata.json')
    assert metadata['arm'] == 'T1' and not metadata['test_only']
    write(dest/'results.json', {'models': means, 'T1_minus_baseline': delta,
        'snapshot': str(source), 'model_sha256': sha256_file(source/'model.pt'),
        'protocol_sha256': sha256_file(ev/'protocol.json'), 'visual_review_complete': False})
    base, _, _ = settings(root, 'T1')
    folders = {'T1-159375': ev/'T1-159375',
               'OLD-239063': Path(protocol['references']['OLD-239063']['folder'])}
    manifests = {k: read(v/'decode_manifest.json') for k, v in folders.items()}
    assert len(protocol['visual_ids']) == 32
    with h5py.File(base.paths['stimuli'], 'r') as h:
        for image_id in protocol['visual_ids']:
            j = protocol['ids'].index(image_id)
            tiles = [('GT', Image.fromarray(h['imgBrick'][image_id]))]
            for label in ('OLD-239063', 'T1-159375'):
                record = manifests[label]['records'][j]
                assert record['image_id'] == image_id
                for c in range(2):
                    f = record['files'][c]
                    assert f['candidate_index'] == c
                    path = folders[label]/f['path']
                    assert sha256_file(path) == f['sha256']
                    with Image.open(path) as im:
                        tiles.append((f'{label} cand{c}', im.convert('RGB').copy()))
            canvas = Image.new('RGB', (1600, 344), 'white')
            draw = ImageDraw.Draw(canvas)
            for n, (title, im) in enumerate(tiles):
                draw.text((n*320+4, 4), title, fill='black')
                canvas.paste(im.resize((320, 320)), (n*320, 24))
            canvas.save(gallery/f'{image_id}.jpg', quality=95)
    keys = [*metrics, 'semantic_score']
    lines = ['# T1 300 epochs 优先评价', '',
        '仅比较新T1-159375与旧159375、旧239063。原500张内部验证图，每图两个固定候选分别评分再图内平均；50步、guidance=4，不使用选图。非独立测试，不改变六个新权重的最终选优规则。', '',
        '| 指标 | 新T1 | 旧159375 | 旧239063 | T1−旧159375 | T1−旧239063 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for k in keys:
        values = [means[l][k] for l in LABELS]+[delta[l][k] for l in LABELS[1:]]
        lines.append('| '+k+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', '语义综合分为AlexNet5、Inception、CLIP identification百分制均值。EfficientNet和SwAV相关距离越低越好，其他指标越高越好。差值均为新减旧；百分制指标差值单位为百分点。', '',
        '## 固定32图', 'GT后依次为旧239063两个候选、新T1两个候选，未按效果筛选。图册生成不代表已查看，内容结论待实际AI视觉检查补充。']
    for i in protocol['visual_ids']:
        lines += ['', f'### 图像 {i}', f'![双候选对照](gallery/{i}.jpg)']
    (dest/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run(root):
    import fcntl
    frozen = root/'runtime/retrain-lr-v1-final'
    out = root/'runs/experiments/retrain-lr-v1'
    dest = out/'priority-T1'
    dest.mkdir(exist_ok=True)
    lock = (dest/'priority.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (dest/'status.json').exists():
        raise RuntimeError('Existing priority attempt; inspect before restarting')
    p = read(out/'pipeline.json')
    assert p['stage'] == 'train-T2' and p['status'] == 'running'
    parent, child = p['pid'], p['stages'][-1]['pid']
    assert str(frozen/'scripts/run_retrain_lr_suite.py') in Path(f'/proc/{parent}/cmdline').read_text()
    assert int(proc(child)[1]) == parent and proc(child)[0] != 'Z'
    state = {'pid': os.getpid(), 'controller_pid': parent, 'T2_torchrun_pid': child,
             'status': 'waiting_T2', 'stages': [], 'training_source_unchanged': p['source']['commit']}
    state['priority_source_commit'] = os.environ['PRIORITY_SOURCE_COMMIT']
    write(dest/'status.json', state)
    replaced = False
    child_eval = None
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'priority coordinator received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    try:
        deadline = time.monotonic()+48*3600
        while True:
            pipeline = read(out/'pipeline.json')
            stage = next(s for s in pipeline['stages'] if s['stage'] == 'train-T2')
            if stage['exit_code'] is not None:
                assert stage['exit_code'] == 0, 'T2 failed; no evaluation takeover'
                break
            if time.monotonic() > deadline:
                raise TimeoutError('T2 wait exceeded 48 hours')
            time.sleep(2)
        state['T2_exit_code'] = 0
        s = read(out/'T2/status.json')
        assert s['status'] == 'completed' and s['completed_updates'] == 159375
        assert not Path(f'/proc/{child}').exists(), 'T2 torchrun must already be reaped'
        assert str(frozen/'scripts/run_retrain_lr_suite.py') in Path(f'/proc/{parent}/cmdline').read_text()
        # Only replace the supervisor AFTER it has recorded a successful T2 exit.
        os.kill(parent, signal.SIGTERM)
        replaced = True
        for _ in range(120):
            if not Path(f'/proc/{parent}').exists():
                break
            time.sleep(.5)
        assert not Path(f'/proc/{parent}').exists(), 'old supervisor still alive'
        interrupted = []
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args = path.read_bytes().decode().split('\0')
                if str(frozen/'scripts/evaluate_retrain_lr.py') not in args:
                    continue
                assert args[args.index('--root')+1] == str(root)
                pid = int(path.parent.name)
                os.kill(pid, signal.SIGTERM)
                interrupted.append(pid)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        state['interrupted_evaluation_pids'] = interrupted
        for pid in interrupted:
            for _ in range(120):
                if not Path(f'/proc/{pid}').exists() or proc(pid)[0] == 'Z':
                    break
                time.sleep(.5)
            assert not Path(f'/proc/{pid}').exists() or proc(pid)[0] == 'Z'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1',
            PYTHONPATH=str(root/'runtime/subject01-4090-1a1fcfa/src'),
            CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        for label in LABELS:
            for phase in ('decode', 'score'):
                busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
                assert not busy, f'GPU occupied; not preempting: {busy}'
                command = [sys.executable, str(frozen/'scripts/evaluate_retrain_lr.py'), '--root', str(root), '--phase', phase, '--label', label]
                item = {'label': label, 'phase': phase, 'command': command, 'exit_code': None}
                state['stages'].append(item)
                state['status'] = phase+'-'+label
                write(dest/'status.json', state)
                with (dest/f'{phase}-{label}.log').open('a') as log:
                    child_eval = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    item['pid'] = child_eval.pid
                    write(dest/'status.json', state)
                    item['exit_code'] = child_eval.wait()
                write(dest/'status.json', state)
                assert item['exit_code'] == 0, item
        report(root, frozen, out)
        state['status'] = 'T1_results_ready_visual_review_pending'
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        if child_eval is not None and child_eval.poll() is None:
            child_eval.wait()
        if replaced:
            # Same frozen controller skips both completed training arms.
            command = ['env', 'PYTHONPATH='+str(root/'runtime/subject01-4090-1a1fcfa/src'),
                'CUDA_VISIBLE_DEVICES=0,1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
                'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'PYTHONUNBUFFERED=1',
                sys.executable, str(frozen/'scripts/run_retrain_lr_suite.py'), '--root', str(root)]
            shell = shlex.join(command)+' >> '+shlex.quote(str(dest/'resumed-controller.log'))+' 2>&1'
            subprocess.run(['tmux', 'new-session', '-d', '-s', 'neuroadapter-retrain-lr-resumed', shell], check=True)
            state['original_controller_resumed'] = True
        write(dest/'status.json', state)
        from datetime import datetime, timezone
        with (root/'repo/EXPERIMENT_LOG.md').open('a', encoding='utf-8') as log:
            log.write(f'\n\n### T1优先评价 {datetime.now(timezone.utc).isoformat()}\n\n'
                      f'状态：{state["status"]}。原控制器已恢复：{state.get("original_controller_resumed", False)}。'
                      '完整命令、PID、退出码和错误（如有）见runs/experiments/retrain-lr-v1/priority-T1/status.json。'
                      '图册生成不等于实际视觉检查，不改变最终选优规则。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
