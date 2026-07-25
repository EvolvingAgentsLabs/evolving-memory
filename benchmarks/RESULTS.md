# Dual-embedding resolver — first measurement

Run 2026-07-25. Gemini embeddings, applicability text written by
`gemini-2.5-flash`, 10 components across 6 domains, 10 tasks each phrased as a
problem rather than as a capability.

## The result

| `task_weight` | What it means | acc@1 | MRR |
|---:|---|---:|---:|
| 0.0 | description-matching — the baseline | **80%** | 0.900 |
| 0.5 | both, evenly | **80%** | 0.900 |
| 1.0 | applicability only | **80%** | 0.900 |

**No difference.** On this corpus, indexing what a component is *for* retrieves
exactly as well as indexing what it *is*, and no better.

## It is not a plumbing bug

The obvious explanation would be that the applicability text is a paraphrase of
the description, so the second embedding carries no new information. Checked:

```
cosine(content, applicability) = 0.753
```

Distinct axis. The generated text reads *"When you need to ensure your electrical
designs are robust, compliant, and optimized for safety…"* against a description
of *"Parses IEC 60364 conductor ampacity and derating tables."* The mechanism
works. It just does not change the answer here.

## What the residual errors actually are

Both misses at every setting are **within-domain confusions**:

```
check whether the wiring in this flat is safe…  → rcd      (wanted cable)
we paid this supplier twice and need to prove…  → ledger   (wanted invoice)
```

The resolver picks the right domain and the wrong tool inside it. Applicability
text does not help with that, because two electrical tools have nearly the same
applicability — both are *for* making wiring safe. The dual embedding was
designed to close a **vocabulary gap between task language and implementation
language**, and these failures are not that. They are disambiguation.

## What this means for the idea

The design dates from 2025, when this gap was wide. A modern embedding model
appears to close much of it on its own: description-matching already reaches 80%
on tasks that share almost no surface vocabulary with the descriptions
("the books do not balance at month end" → a double-entry reconciler).

So the honest statement is narrower than the one this work started from:

> Recovering the dual-embedding resolver did **not** reproduce a measurable
> advantage over description-matching on a small corpus with a strong embedding
> model. Whether it helps at all is now an open question with a number attached,
> not an assumption.

## Before anyone builds on this

- **n = 10 components, 10 tasks.** Far too small to conclude much. Both arms
  scoring 80% could be the corpus being easy rather than the arms being equal.
- **One embedding model.** The hypothesis that modern embeddings closed the gap
  predicts the advantage *returns* on a smaller or older encoder. Untested, and
  it is the most informative next run — especially given this organisation cares
  about small local models, where the gap is most likely to still exist.
- **The corpus was written by the same person as the method.** Independently
  sourced tasks would be worth more than more of these.

## Reproducing

```bash
GEMINI_API_KEY=... python benchmarks/resolver_benchmark.py
```

Without a key it falls back to template-written applicability, which is weaker
and will understate any difference.
