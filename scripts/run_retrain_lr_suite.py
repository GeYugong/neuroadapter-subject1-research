"""Persistent, idempotent T1 -> T2 -> fixed evaluation controller."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

from retrain_lr_core import U,NODES,REPO,output,code_identity,latest_checkpoint,verify_cached_assets
from neuroadapter_research.atomic import write_json_atomic
from neuroadapter_research.checkpoint import verify_checkpoint,verify_inference_snapshot


def available_gpus():
    names=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip().splitlines()
    if len(names)!=2 or any('RTX 4090' not in name for name in names): raise RuntimeError(f'unexpected GPUs: {names}')
    tasks=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).strip()
    if tasks: raise RuntimeError(f'GPUs are in use; no preemption allowed: {tasks}')
    return names


def run(root):
    out=output(root)
    lock=(out/'suite.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=code_identity(root);verify_cached_assets(root)
    gate=json.loads((out/'preflight.json').read_text());assert gate['passed'] and gate['source']==identity
    assert json.loads((out/'evaluation/replay.json').read_text())['passed']
    assert json.loads((out/'evaluation/smoke-summary.json').read_text())['image_count']==2
    path=out/'pipeline.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    if previous:
        assert previous['source']==identity,'cannot resume controller with another source'
        if previous['status']=='trained_evaluated_exported': print('Already exported; no retraining');return
    state={**previous,'source':identity,'pid':os.getpid(),'status':'running','started_at':datetime.now(timezone.utc).isoformat()}
    state.setdefault('stages',[])
    env=dict(os.environ,PYTHONPATH=str(root/'runtime/subject01-4090-1a1fcfa/src'),CUDA_VISIBLE_DEVICES='0,1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')

    def note(message):
        with (root/'repo/EXPERIMENT_LOG.md').open('a',encoding='utf-8') as f:
            f.write(f'\n\n### retrain-lr-v1 自动记录 {datetime.now(timezone.utc).isoformat()}\n\n{message}\n')

    def execute(stage,args,gpus='0,1'):
        assert code_identity(root)==identity
        state.update(stage=stage,gpus=available_gpus())
        record={'stage':stage,'command':[sys.executable,*args],'exit_code':None,'start':datetime.now(timezone.utc).isoformat()}
        state['stages'].append(record);write_json_atomic(path,state)
        note(f'开始 `{stage}`，源码 `{identity["commit"]}`，可见GPU `{gpus}`；完整命令记录pipeline.json。')
        with (out/f'{stage}.log').open('a') as log:
            process=subprocess.Popen(record['command'],stdout=log,stderr=subprocess.STDOUT,env=dict(env,CUDA_VISIBLE_DEVICES=gpus))
            record['pid']=process.pid;write_json_atomic(path,state)
            rc=process.wait()
        record.update(exit_code=rc,end=datetime.now(timezone.utc).isoformat());write_json_atomic(path,state)
        note(f'`{stage}` 退出码 `{rc}`。')
        if rc: raise RuntimeError(f'{stage} exited {rc}; see {stage}.log')

    try:
        for arm in ('T1','T2'):
            folder=out/arm;status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
            if status.get('status')!='completed':
                args=['-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(REPO/'scripts/train_retrain_lr.py'),'--root',str(root),'--arm',arm]
                if folder.exists():
                    points=[latest_checkpoint(folder/k) for k in ('checkpoints','milestones')]
                    point=max((p for p in points if p),key=lambda p:p.name,default=None)
                    if point is None: raise RuntimeError(f'{arm} has partial output but no complete recovery point; inspect before continuing')
                    args+=['--resume',str(point)]
                execute('train-'+arm,args)
            status=json.loads((folder/'status.json').read_text())
            assert status['status']=='completed' and status['completed_updates']==U,f'{arm} did not complete; not starting next arm'
            for u in NODES:
                verify_inference_snapshot(folder/'snapshots'/f'snapshot-update-{u:08d}')
                verify_checkpoint(folder/'milestones'/f'checkpoint-update-{u:08d}',expected_world_size=2)
        # This check excludes gradients/weights, which legitimately differ after update1.
        for rank in range(2):
            assert (out/'T1'/f'random_trace_rank{rank}.jsonl').read_bytes()==(out/'T2'/f'random_trace_rank{rank}.jsonl').read_bytes()
        from evaluate_retrain_lr import labels
        protocol=json.loads((out/'evaluation/protocol.json').read_text())
        for label in labels():
            if label.startswith('OLD') and not protocol['references'][label]['available']: continue
            execute('decode-'+label,[str(REPO/'scripts/evaluate_retrain_lr.py'),'--root',str(root),'--phase','decode','--label',label],'0')
            execute('score-'+label,[str(REPO/'scripts/evaluate_retrain_lr.py'),'--root',str(root),'--phase','score','--label',label],'0')
        execute('export',[str(REPO/'scripts/summarize_retrain_lr.py'),'--root',str(root)],'0')
        state.update(status='trained_evaluated_exported',stage='fixed32_visual_review_pending')
        note('两条159375更新训练、六份新权重完整500图评价、同条件旧基线比较和最佳新候选导出完成。32图册已生成，仍须实际打开作定性检查；不再训练、不自动认定可作ROI结论。')
    except BaseException as exc:
        state.update(status='failed_or_interrupted',error=repr(exc))
        note(f'控制器停止于 `{state.get("stage")}`：`{exc!r}`。不启动后续阶段、不改超参数；使用同一冻结入口恢复。')
        raise
    finally:
        write_json_atomic(path,state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
