# Data

Large files are not committed. Rebuild them with the scripts in `scripts/`.

```
data/raw/         downloads: ConvoKit CMV, args.me v1.0-cleaned, IBM-ArgQ-Rank-30k
data/processed/   dialogues, training examples, passages (JSONL)
data/indexes/     BM25 and FAISS indexes
data/toy/         tiny hand-written files for the --dummy runs and the tests (committed)
```

Datasets (from the proposal):

| Dataset | Role | Size |
| --- | --- | --- |
| CMV Winning Arguments (Tan et al., 2016) | Trains the generator, supplies test dialogues | 293,297 comments, 3,051 threads |
| args.me v1.0-cleaned (Ajjour et al., 2019) | Main retrieval index | 382,545 arguments |
| IBM-ArgQ-Rank-30k (Gretz et al., 2020) | Backup index and retrieval-stance test | about 30,497 arguments, 71 motions |

Fill in exact download links and file names in `scripts/download_data.py` as you confirm them. The professor accepts a documented download script for large public data.
