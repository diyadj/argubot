# Adapting CS425_Project_Base.ipynb for ArguBot

For Andrea. The base notebook was written for a GPT-2 emotion bot. Most of its logic now lives in `src/argubot`, tested, so the notebooks shrink to thin runners that import from the package. Aim for four short notebooks (the list is at the bottom).

## Do this first (by 19 Oct)

1. Load `Qwen/Qwen3.5-0.8B-Base` with the transformers version Colab gives you. If it will not load, decide on the fallback (`gpt2-medium`) and tell the group the same day. The fallback changes the LoRA target modules (`c_attn`) and the context limit (1024 tokens).
2. List the model's linear layers and confirm `target_modules="all-linear"` reaches the attention and Gated DeltaNet projections.
3. Run one tiny LoRA step on the toy data to check memory and fp16 on a T4. If the loss is NaN, load the base model in fp32 or move to the SCIS GPU.
4. Write down the trainable parameter count and percentage, and the seconds per step. The report asks for both.

## Cell by cell (numbers follow the base notebook)

| Cells | What the base notebook does | What to change |
| --- | --- | --- |
| 0 | Intro for "any dialogue idea" | Rewrite for ArguBot: claim log, retrieval, five systems. |
| 1 | `pip install` list | Clone the repo, then `pip install -e .` and `pip install -r requirements.txt`. Add `python -m spacy download en_core_web_sm`. Drop `evaluate` if unused. Pin the versions that worked. |
| 2 to 3 | A `CONFIG` dict with GPT-2, r=8, 256 tokens | Delete it and call `load_config("configs/config.yaml")`. Remove `signal_labels`, `use_classifier` and `use_retrieval`. Keep the device line and use `transformers.set_seed(cfg["seed"])`. |
| 4 to 5 | `load_dataset` placeholder | Replace with reading `train_examples.jsonl` and `val_examples.jsonl` (Tricia and Diya produce them) through `argubot.io.read_jsonl`. |
| 6 to 7 | `format_example`, `tokenize_batch` | Delete both. Use `ReplyOnlyDataset` from `argubot.generate.training`, which calls the shared `build_prompt` and masks the prompt in the labels. Only set `pad_token` if the tokenizer has none. |
| 8 to 9 | BERT signal classifier | Delete. ArguBot has no detection step. |
| 10 to 11 | TF-IDF and dense retrieval in the notebook | Delete. Retrieval lives in `argubot.retrieval` (Diya). The training data already carries the retrieved passages. |
| 12 to 13 | LoRA on `c_attn`, r=8, `.to(device)` | Use `build_lora_config(cfg)` (r=16, alpha=32, all-linear). Let the Trainer place the model. Print the trainable parameter line and keep it. |
| 14 to 15 | `DataCollatorForLanguageModeling`, Trainer commented out | Use `ReplyOnlyCollator`. Switch the Trainer on with a validation set, evaluation each epoch, gradient accumulation and warmup from the config. Save the adapter only, to `cfg["paths"]["adapter_dir"]`. |
| 16 to 17 | `perplexity`, `distinct_n`, placeholder task metric | Use `perplexity` and `distinct_n` from `argubot.eval.metrics`. The task metric is `contradiction_rate` (Cassey), run from `scripts/evaluate.py`. |
| 18 to 19 | `generate_response` plus an `input()` loop | Delete. Evaluation goes through `scripts/run_systems.py`. Keep a separate demo notebook (below). |

## Easy to get wrong (found while reading the base notebook)

- **EOS gets masked.** `DataCollatorForLanguageModeling` masks every pad token, and the notebook sets pad to EOS, so the model never learns to stop. `ReplyOnlyCollator` fixes this.
- **Training and generation prompts differ.** `format_example` has no evidence slot but `generate_response` adds one. Everything must go through `build_prompt`.
- **The loss covers the prompt.** The proposal says reply tokens only.
- **Slot blanking must be redrawn every epoch.** If it runs once during preprocessing, each example keeps the same blanks forever. `ReplyOnlyDataset` redraws them on each read.
- **Truncation side.** The default truncates the right side, which cuts off the `[response]` tag on long prompts. `HFGenerator` sets left truncation.
- **`max_new_tokens=60` is too short** for replies of about 150 tokens. It comes from `decoding.max_new_tokens` now.
- **`padding="max_length"` wastes compute.** Pad per batch in the collator instead.
- **fp16 plus PEFT** can raise "Attempting to unscale FP16 gradients". Load the base in fp32, or cast the LoRA parameters to fp32.
- **Never tune on the test split.** Hyperparameters are tuned on validation, and the report says so.
- **Perplexity** in the base notebook counts pad tokens and the prompt. Use the repo version, and say in the report which slots were filled when it was measured.

## Things to record for the report as you go

Model and adapter size, trainable parameters, training subset size, epochs, learning rate search (what you tried, what won), GPU, seconds per epoch, and any NaN or out-of-memory problems and how you fixed them.

## Target notebooks

1. `01_smoke_test.ipynb`: load the model, one LoRA step on toy data, generate one reply.
2. `02_train.ipynb`: load config and examples, train, save the adapter, print validation perplexity.
3. `03_run_and_evaluate.ipynb`: run `scripts/run_systems.py` and `scripts/evaluate.py` for all five systems, show the results table.
4. `04_demo.ipynb`: an interactive loop for 10 Nov that builds one `ArguBotPipeline`, keeps one `ClaimTracker` alive across turns, and prints the claim log and the retrieved passages next to each reply, so the audience can see the mechanism.
