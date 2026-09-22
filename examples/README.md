# Examples

Ready to run ASCEC input files. Each one is the exact `.asc` that produced the
results quoted in the User Manual, so running it reproduces those numbers (up to
the random seed — see *Reproducibility* in the manual to pin one).

Run any of them from its own directory:

```bash
cd w6/rigorous && ascec w6.asc
```

## The three modes

| Mode | Protocol | What you get |
|---|---|---|
| `preliminary` | annealing → opt → cosmic | A fast survey. No frequencies, so the structures are not guaranteed to be true minima. |
| `rigorous` | adds `ref` (Opt+Freq) → cosmic | Verified minima and Boltzmann populations. |
| `ultimate` | adds `eref` (single point) → cosmic | Rigorous geometries with high level energies. |

## What is here

| Input | System | Mode | Wall time | Result |
|---|---|---|---|---|
| `glyw2/preliminary/glyw2.asc` | glycolaldehyde + 2 H₂O | preliminary | 55 s | 108 accepted → 40 motifs |
| `glyw2/rigorous/glyw2.asc` | glycolaldehyde + 2 H₂O | rigorous | 1 d 11 h | 269 accepted → 66 → 44 minima |
| `w6/preliminary/w6.asc` | water hexamer | preliminary | 3 min | 78 accepted → 19 motifs |
| `w6/rigorous/w6.asc` | water hexamer | rigorous | 1 d 3 h | 215 accepted → 39 → 22 minima |
| `w6/ultimate/w6.asc` | water hexamer | ultimate | 4 d 14 h | 259 accepted → 52 → 29 → 26 minima |
| `cis_formic/preliminary/cis_formic.asc` | *cis* formic acid dimer | preliminary | 1 min | 101 accepted → 23 motifs |
| `cis_formic/rigorous/cis_formic.asc` | *cis* formic acid dimer | rigorous | 10 h 43 m | 574 accepted → 46 → 21 minima |
| `li5/preliminary/li5_pre.asc` | Li₅ atomic cluster | preliminary | 3 min | 41 accepted → 6 motifs |
| `li5/ultimate/li5.asc` | Li₅ atomic cluster | ultimate | 4 h 14 m | 192 accepted → 12 → 4 → 4 minima |

Wall times are from the machines these were validated on and depend heavily on
your core count and on the quantum chemistry backend. The `preliminary` runs stay
at the GFN2-xTB level throughout and finish in minutes; the `rigorous` and
`ultimate` runs spend almost all of their time in the DFT and coupled cluster
stages.

## Notes on particular inputs

- **`li5`** is an atomic cluster. Every atom counts as its own molecule, so line 13
  reads `5` for five lithium atoms and each block below it holds a single atom.
  It also carries a non singlet multiplicity (`0 2` on line 12) and `uhf 1` in the
  xTB template. The rotation field on line 7 is left at `1.0`: rotating a lone atom
  does nothing, so the value is simply ignored for monatomic species.
- **`glyw2`** has both configurational and conformational freedom, so conformational
  sampling is switched on with a nonzero percentage on line 8.
- **`w6/ultimate`** runs the full three tier pipeline and includes a
  DLPNO-CCSD(T) energy refinement. It is the longest example here.

Each input carries its own `#Protocol` block and embedded QM templates at the
bottom of the file, so nothing else is needed to run it. Adjust `nprocs` in those
templates to match your machine before launching a long job.
