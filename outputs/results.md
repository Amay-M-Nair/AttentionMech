# Latin -> English, test results

Decoded once with beam 8, length penalty 1.5, chosen on 500 validation sentences. Checkpoint from epoch 60.

| | BLEU | chrF |
|---|---|---|
| copy baseline | 0.31 | 19.74 |
| model | 18.87 | 42.78 |

| source tokens | n | copy | BLEU | chrF |
|---|---|---|---|---|
| 1-20 | 729 | 0.74 | 26.01 | 46.38 |
| 21-35 | 1019 | 0.39 | 25.89 | 47.13 |
| 36-55 | 733 | 0.11 | 18.84 | 42.90 |
| 56+ | 522 | 0.17 | 11.04 | 38.76 |

| split | n | copy | BLEU | chrF |
|---|---|---|---|---|
| Vulgate | 984 | 0.45 | 40.37 | 61.00 |
| classical | 2019 | 0.19 | 10.36 | 37.16 |
