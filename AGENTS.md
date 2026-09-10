# Project Instructions

- Respond in Simplified Chinese. Use English conventional commit messages.
- The active training host is the SSH alias `neuroadapter-4090` (also `双卡4090`), with two RTX 4090 GPUs. Do not launch training or inference on the former RTX 5090 host; migration reads, project-local migration artifacts, and revocation of temporary transfer authorization are the only exceptions.
- Read local SSH configuration for connection details; never put passwords or private keys in this repository, logs, or GitHub.
- Create and modify remote project files only below `/data1/matengyu/geyugong/neuroadapter-subject1-research/`. Do not change shared environments or stop other users' processes.
- The repository is the `repo/` child of that project root. Data, environments, model assets, and runs are siblings, not Git content.
- Keep the GitHub repository public. Preserve historical experiment records and append each operation to `EXPERIMENT_LOG.md` in Chinese.
- Formal training requires the configured integrity, hardware, batch, recovery, decode, and evaluator gates. Never manufacture approval or relabel a gate checkpoint as a formal checkpoint.
- Freeze one batch geometry before selection. Selection and final must retain the same method and global batch size 16. Do not use standard-test results to choose training settings.
- Decision on 2026-09-10 supersedes the earlier final-retraining plan: select one existing selection snapshot, report it, and stop. Do not launch additional training, resume training, or the 9000-image final retraining without a new explicit user request. Preserve the selected model's real 8500-image training provenance.
- Hugging Face uploads must use the project-local `credentials/hf-upload` credential store and verify account `gugabobo` before writes. Never use the server home-directory login. The user authorized public weight backups, not publication of datasets or credentials.
