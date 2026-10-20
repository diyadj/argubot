# Runbook: test the smoke-test notebook on the SCIS GPU cluster

Step-by-step for running `notebooks/01_smoke_test.ipynb` on the Origami GPU cluster
(violet.scis.dev). Follow it top to bottom the first time; after that, only steps 5–9 repeat.
For deeper reference (sbatch training, partitions, etc.) see [SCIS_VIOLET.md](SCIS_VIOLET.md).

**What success looks like:** the notebook loads `Qwen/Qwen3.5-0.8B-Base` on a GPU, runs a few LoRA
steps on toy data with a finite, decreasing loss, and prints a summary with the trainable-parameter
count and seconds/step. It takes ~10–20 min end to end (most of it a one-time model download).

---

## One-time prerequisites (do these once, before step 1)

- [ ] **Cluster account** — request it from the professor (comes with the SMU VPN account).
- [ ] **SMU VPN connected** if you are off campus. Without it, the cluster is unreachable.
- [ ] **ClearPass healthy**, and no custom DNS (e.g. `1.1.1.1` / `8.8.8.8`) set on your network adapter.
- [ ] **GitHub access** to `diyadj/argubot` plus a **Personal Access Token (classic)** with the `repo`
      scope — create at github.com → Settings → Developer settings → Personal access tokens. You'll
      paste it as the "password" when cloning. (The repo is private, so your GitHub password won't work.)

---

## Steps

### 1. Log in to the cluster
In **PowerShell** (not Command Prompt), using your own account name:
```bash
ssh <yourAccount>@10.193.104.102
```
We log in by **IP** because the hostname `origami.smu.edu.sg` often won't resolve off-campus. First
login forces a password change (type the temp password, then a new one twice; it logs you out, then
`ssh` back in with the new password).

### 2. Get the code (one-time clone)
```bash
cd ~ && git clone https://github.com/diyadj/argubot.git
```
This clones the default `main` branch. When prompted: **username** = your GitHub username,
**password** = your **personal access token**.
> Already cloned from a previous session? Just update instead: `cd ~/argubot && git pull`

### 3. Launch a GPU notebook
```bash
mynotebook
```
Answer the prompts:
- **SMU email** → e.g. `yourname.2023@scis.smu.edu.sg` (for start/stop notifications)
- **Notebook environment** → `2) PyTorch`
- **GPU** → choose the **GPU-enabled** option
- **GPU type** → `1) any` (fastest to schedule; any of 3090/a40/l40s handles this fine)

It sets up Jupyter, queues the job, and when ready prints a **STEP 1 / STEP 2** block with an
`ssh -N -L ...` tunnel command and a `http://127.0.0.1:<port>/...token...` URL. Leave this terminal open.

### 4. Open the SSH tunnel (on your own computer)
Open a **second** PowerShell on your laptop and paste the `ssh -N -L ...` line exactly as `mynotebook`
printed it, e.g.:
```bash
ssh -N -L 8997:aloha:8997 <yourAccount>@10.193.104.102 -v
```
Enter your cluster password. It prints some lines then goes quiet — **that means it's working**. Do
not close it. (If it fails on the hostname, swap in `10.193.104.102`.)

### 5. Open the notebook in your browser
Paste the full `http://127.0.0.1:<port>/...token...` URL into your browser, then open
`argubot/notebooks/01_smoke_test.ipynb`.

### 6. Run all cells
**Kernel → Restart Kernel and Run All Cells** (or Run All). The first cell handles everything
automatically: it moves to the repo root, puts `src/` on the path, and installs
`torch`/`transformers`/`peft`/`accelerate` if missing. No manual edits needed.
> First run downloads the model (~1.5 GB) — it may sit for several minutes, then caches under
> `~/.cache/huggingface`, so later runs are instant.

### 7. Check it worked
- Config cell prints `device: cuda` with a GPU name (e.g. `NVIDIA GeForce RTX 3090`).
- The LoRA-step cell prints a **finite, decreasing loss** (no `NaN`).
- The summary cell prints the trainable-parameter % and seconds/step.

### 8. Record for the report
Copy the summary cell's output. A good run looks like:
```
base model       : Qwen/Qwen3.5-0.8B-Base  (model_type=qwen3_5_text)
device           : cuda / NVIDIA GeForce RTX 3090
dtype / fp16     : torch.float16, USE_FP16=True
LoRA             : r=16, alpha=32, target=all-linear
trainable params : 10,822,656 / 763,215,680 = 1.418%
seconds/step     : 0.51s  (batch=4)
```

### 9. Free the GPU when done
Your QOS allows only **1 running job**, so always release it. From any terminal:
```bash
ssh <yourAccount>@10.193.104.102 scancel <jobID>
```
(The `<jobID>` is in the `mynotebook` output and from `myqueue`.) The session also auto-ends after 6 h.

---

## Run the chat demo on the cluster

The demo page and the bot run together on the GPU node; you open the page on your laptop through a second tunnel.

1. Start a GPU notebook (steps 1–5 above). The built passages and indexes are already linked in `~/argubot`
   (see `docs/DATA_NOTES.md`, "Built files on the SCIS cluster").
2. In a Jupyter terminal on the node:
   ```bash
   cd ~/argubot
   python scripts/serve_demo.py --dummy --host 0.0.0.0 --port 8000   # fake replies, checks the page
   python scripts/serve_demo.py --host 0.0.0.0 --port 8000           # real bot, needs outputs/adapter
   ```
   `--host 0.0.0.0` lets the login node forward to it; the default `127.0.0.1` only answers on the node itself.
3. On your laptop, in another PowerShell, tunnel the port to the node (the node name is in the `mynotebook`
   tunnel line, e.g. `lucas`):
   ```bash
   ssh -N -L 8000:<node>:8000 <yourAccount>@10.193.104.102
   ```
4. Open http://127.0.0.1:8000.

The demo only runs while the GPU job does (6 h for `mynotebook`). On presentation day, start a fresh job
well before you present.

## Troubleshooting (the exact errors we hit)

| Symptom | Cause / fix |
| --- | --- |
| `ssh: Could not resolve hostname origami.smu.edu.sg` | VPN not connected, or you used the hostname. Connect the SMU VPN and use the IP `10.193.104.102`. |
| `ModuleNotFoundError: No module named 'google'` (setup cell) | You're on an **old** copy of the notebook. `cd ~/argubot && git pull` to get the fixed setup cell. |
| `ModuleNotFoundError: No module named 'argubot'` (config cell) | Same — pull the latest notebook. (An editable install isn't visible to a running kernel; the fixed setup cell puts `src/` on the path instead.) |
| `Warning: ... unauthenticated requests to the HF Hub` | Harmless. Public models download fine; a `HF_TOKEN` only raises rate limits. |
| `causal_conv1d` / `flash-linear-attention` ... `falling back to reference` | Harmless. Qwen3.5's Gated DeltaNet runs a correct, slightly slower reference kernel. No action needed. |
| Model download crawling | The node's bandwidth. It caches after the first run, so it's a one-time cost. |
| `device: cpu` instead of `cuda` | You picked a CPU notebook in step 3, or the GPU wasn't allocated. `scancel` and relaunch choosing the GPU option. |

## Unattended alternative (optional)
To run the notebook without keeping a terminal open (up to 24 h), use `execnotebook` instead of
`mynotebook` — the `.ipynb` must be in your home directory. See [SCIS_VIOLET.md](SCIS_VIOLET.md).
