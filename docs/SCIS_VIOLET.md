# Running ArguBot on the SCIS Origami GPU cluster (violet.scis.dev)

> **Just want to run the smoke-test notebook?** Follow the step-by-step runbook in
> [RUN_ON_VIOLET.md](RUN_ON_VIOLET.md). This page is the deeper reference (sbatch training,
> partitions, file transfer).

Use this when Colab's T4 is not enough (fp16 NaNs, out-of-memory, or a long training run).
Docs: https://violet.scis.dev/docs

**Two paths:**
- **Notebooks (recommended for `01_smoke_test` and `02_train`):** run the `.ipynb` directly on a GPU
  with the cluster's `mynotebook` (interactive, 6 h) or `execnotebook` (unattended, 24 h) commands.
- **Headless training:** submit [`scripts/violet_train.sbatch`](../scripts/violet_train.sbatch) with `sbatch` once `train()` is implemented.

---

## 0. Prerequisites

- A cluster account — **request it from the professor/instructor** (same as the VPN account).
- A **healthy ClearPass** (download from IITS). If you set a custom DNS like `1.1.1.1` / `8.8.8.8`, remove it.
- **SMU VPN** if you are off the SMU network (which you are, at home).
- Resource list (GPU models/tags) is in a Google Sheet linked from the docs; sign in with `you@smu.edu.sg`.

## 1. Log in

From PowerShell (not Command Prompt). The hostname `origami.smu.edu.sg` only resolves on SMU's
internal DNS, which may not work off-campus even on the VPN — so **log in by the login-node IP**,
which is what actually works here:

```bash
ssh CS425G3@10.193.104.102
```

(`10.193.104.102` is the origami login node. `ssh CS425G3@origami.smu.edu.sg` works too *if* the
hostname resolves for you.) First login forces a password change: enter the temporary password, then
a new one; you'll be logged out, so `ssh` in again with the new password. Change it later with `passwd`.

> If `ssh` says `Could not resolve hostname`, you used the hostname and DNS failed — use the IP.

## 2. Your quota values (CS425 G3)

```bash
myinfo
```

The values already filled into [`scripts/violet_train.sbatch`](../scripts/violet_train.sbatch):

| Field | Value |
| --- | --- |
| Account | `cs425` |
| Partition | `project` |
| QOS | `cs425qos` |
| Limits | 1 GPU, 30 CPUs, up to 100 GB RAM, 1-day max runtime |
| Concurrency | 1 running job, 2 submitted jobs at a time |
| Home quota | 180 GB |
| Scratch dir | `/common/scratch/projectgrps/CS425/CS425G3/` |

## 3. Put the code on the cluster

Git is recommended:

```bash
git clone <your argubot repo URL>
cd argubot
```

(Alternatively `scp -r` the folder up, or use WinSCP on Windows.)

---

## 4. Path A — run the notebook on a GPU (recommended for the smoke test)

### Interactive: `mynotebook` (up to 6 h)

```bash
mynotebook
```

Answer the prompts: SMU email → **GPU enabled notebook** → **Python v3.11.7** (we install torch
ourselves). It prints a one-off SSH tunnel command and a `127.0.0.1` URL (also saved to
`~/jupyterReconnectingInstructions.txt`). On **your laptop**, open a second PowerShell and run the
tunnel command it gave you:

```bash
ssh -N -L <port>:<node-ip>:<port> CS425G3@10.193.104.102
```

(`mynotebook` prints this line for you with the real port and node — copy it, but if it uses the
hostname and that doesn't resolve, swap in `10.193.104.102`.)

Then open the `http://127.0.0.1:<port>/?token=...` URL in your browser. In the first notebook cell,
install the package (inside the notebook, prefix shell commands with `!`):

```python
!pip install -e .
!pip install -r requirements.txt
!python -m spacy download en_core_web_sm
```

Then run `notebooks/01_smoke_test.ipynb`. When done, free the GPU with `scancel <jobid>` (get the id
from `myqueue`).

> The smoke-test notebook already detects it is **not** on Colab and skips the clone/install cell, so
> the `!pip install` above is all you need.

### Unattended: `execnotebook` (up to 24 h)

For a long run you don't want to babysit. Run `mynotebook` once first (it sets up the jupyter venv),
then:

```bash
execnotebook
```

Give it the notebook **filename** (it must live in your **home directory**, not scratch), your SMU
email, choose **PyTorch**, keep the output as a new file, and request a **GPU**. It queues a job and
writes `notebook.output.<timestamp>.ipynb` when finished. Check progress with `myqueue`.

---

## 5. Path B — headless training with sbatch

Once `train()` in `src/argubot/generate/training.py` is implemented:

1. Edit the four marked lines in [`scripts/violet_train.sbatch`](../scripts/violet_train.sbatch) with your `myinfo`
   values (`--partition`, `--account`, `--qos`, `--mail-user`).
2. Submit it:

```bash
chmod +x scripts/violet_train.sbatch
sbatch scripts/violet_train.sbatch
```

The adapter is saved to `outputs/adapter` (from `configs/config.yaml`). Logs land in
`<USERNAME>.<JOBID>.out` in the directory you ran `sbatch` from. The first run creates a conda env and
installs dependencies, so it is slower; later runs reuse the env.

## 6. Bring results back to your laptop

```bash
scp -r CS425G3@10.193.104.102:~/argubot/outputs/adapter ./outputs/adapter
```

## Useful commands

| Command | What it does |
| --- | --- |
| `myinfo` | Your account, partition, QOS, limits |
| `myqueue` | Your running/pending jobs (and their job ids) |
| `myjob <jobid>` | Detail on a running / recently finished job (resources used) |
| `mypastjob <days>` | Job history for the last N days (max 30) |
| `scancel <jobid>` | Cancel a job / free a notebook GPU |

## Caveats

- **1 GPU, 1-day cap, 1 running job** on `cs425qos` (100 GB RAM is plenty for a 0.8B LoRA). If the
  full training run won't fit in a day, set `training.max_train_examples` in the config, or ask SCIS IT.
- **Preemption:** community-queue jobs can be killed to free resources. For long runs, checkpoint, and
  consider `#SBATCH --requeue` (and `--constraint=nopreempt` if available) in the sbatch script.
- **fp16 NaNs:** the whole reason to come here if the T4 misbehaves — a bigger cluster GPU (a40 / l40s /
  h100) handles fp16 cleanly; or set `training.fp16: false` to run fp32.
- **Pin versions:** after a training run works, paste `pip freeze` output into `requirements.txt` so the
  report's "how to run" section is exact (Qwen3.5 is version-sensitive).
