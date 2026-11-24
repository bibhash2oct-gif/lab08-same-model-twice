# PyTorch or Keras for the weekly bike demand job

For the manager and the two team members. The numbers come from
`reports/comparison.csv`: both frameworks, seeds 42, 1 and 2, the same rows and
the same settings.

## What I observed

1. **It is the same model.** Both networks have 4,801 parameters. With the
   trained PyTorch weights copied into Keras, each kernel transposed, the largest
   prediction gap on the validation rows was 3.1e-05 rentals.
2. **Accuracy is close.** Mean test MAE was 2.52 rentals in PyTorch and 2.64 in
   Keras, against 8.57 for linear regression. PyTorch led on all three seeds, but
   the frameworks start from different weights and shuffle differently, so three
   seeds cannot show whether a 0.12 gap is real.
3. **PyTorch was faster.** A 150-epoch run took 2.3 seconds in PyTorch and 4.7
   in Keras on my MacBook, about twice as long. `fit` adds a fixed cost to every
   epoch, so the gap would shrink on a larger model.
4. **Keras needs less training code.** My PyTorch loop (`run_epoch` and `train`)
   is 61 lines I had to get right, including keeping the best weights by hand. In
   Keras the loop is one `fit` call, and two callbacks do the bookkeeping.
5. **Saving and debugging differ.** Keras saved one 81 KB `.keras` file that
   `load_model` restores in one line; my PyTorch code saves nothing yet. But I
   cannot add a print line inside `fit`: I had to write a callback to show
   progress. And Keras predictions have shape (N, 1), which gives a wrong MAE
   unless flattened.

## Recommendation: PyTorch

- **Hugging Face later in the year.** Transformers v5 supports only PyTorch
  ([Hugging Face](https://huggingface.co/blog/transformers-v5)). Choosing Keras
  now would mean a second migration.
- **A weekly job.** The PyTorch version already works, has 13 tests, and trained
  faster with accuracy at least as good.
- **A small team.** Two people should maintain one framework, and one of them
  already knows PyTorch.

## What we give up

Keras's shorter code: built-in early stopping, checkpoints and one-file saving.
We must maintain our own loop and add checkpoint code, and the Keras teammate has
to learn PyTorch.

## What would change my recommendation

If the team drops the Hugging Face plan and the Keras teammate runs the job alone,
I would choose Keras. The accuracy gap is within what seeds can produce, and the
shorter code is easier for one person to keep running every week.
