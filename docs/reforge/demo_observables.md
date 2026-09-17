# What the declarative observables look like finished

A demonstration branch, not a merge candidate. It exists to answer a question
the team raised about the rewrite: how much does this grow the repository?

The short answer is that the growth is in declarations and tests, the code that
reads them is small, and the tree it is replacing is deleted on the way out.

## The repository today

| area | lines | share |
|---|---|---|
| `tests/` | 55,715 | 62 % |
| `mace/` | 29,976 | 33 % |
| `packages/` | 2,749 | 3 % |
| `docs/` | 1,666 | 2 % |

The whole v1 tree, after five merged tickets, is 3 % of the repository. What
already dominates it is the test suite, and the rewrite does not change that.

`mace/` is not a permanent addition on the other side of the ledger either.
RET-1 through RET-6 delete it capability by capability, and the debt book
tracks each one. The window where both trees exist is the price of having a
live oracle to measure against, and it is bounded by those six tickets.

## What this branch adds

| file | lines | of which code |
|---|---|---|
| `defaults/full_surface.yaml` | 261 | 0, it is data |
| `mace_torch/heads.py` | 177 | **79** |
| the end-to-end test | 297 | |

Seventy-nine lines of code. That is the entire head layer for the entire v0.3
output surface: one readout per declared row, one branch on `per_atom` deciding
whether values are summed into their graph, and one loop applying declared
derivatives with their declared signs.

Nothing in those 79 lines knows what a dipole is, or a polarizability, or a
Born effective charge, or a magnetic force.

## What it is replacing

Not the backbone. The layer above it: readouts, output assembly, and the
derivative plumbing. In the frozen tree that layer is spread across

- **11 model classes** in `mace/modules/{models,extensions}.py`, together 3,787
  lines, each with its own `forward` returning its own dict,
- **472 mentions of `compute_*`** across `mace/`, the booleans that decide
  which of those outputs is produced,
- a ladder of string comparisons on the model class name in `run_train.py`
  that sets six of those booleans,
- and two hand-kept key lists in the ase calculator that classify 22 of the 43
  output names and let the other 21 through with their padding rows intact.

## The three things the test shows

1. **The whole surface comes out of a head that knows none of it.** 28 declared
   observables, each with the shape its irreps string gives it, per-atom or
   per-graph as its row says.
2. **The sign is data.** `forces` and `magforces` are checked against central
   differences of the energy, not against the autograd call that produced them.
   `magforces` is the case that pays for the derivative grammar being written
   over declared inputs rather than over positions and the cell: nothing in the
   head or the engine spells `magmom`.
3. **A new observable is a row.** The test appends an octupole to a YAML string
   and asserts it appears in the output with the right shape, along with its
   position derivative. No import, no subclass, no registry entry, no edit to
   any module.

## What is deliberately not here

The head is a linear map, not an equivariant readout. The demonstration needs
the *shape* to come from the declaration, and it does; a real head reads the
same irreps string to build a real readout.

The design of the head layer is ARCH-3's and is not settled. This branch is a
worked example so the shape is visible before that ticket starts. Expect it to
be replaced rather than merged.
