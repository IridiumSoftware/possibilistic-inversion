# volve - real-data integration on the Equinor Volve walkaway VSP

This subpackage takes the shipped possibilistic-inversion methodology
(`posdec`) from synthetic-only into real-data territory. The target
dataset is the **Equinor Volve walkaway VSP** (North Sea, open release
2018), with the well **15/9-F-1A** sonic log as independent Vp ground
truth.

## Workflow phases

| Phase | Status | What |
|-------|--------|------|
| 1     | landed | Ingestion skeleton: geometry deck + SEG-Y reader + LAS reader, all N=1 smoked on synthetic data. |
| 2     | open   | First-arrival picking on the real walkaway shot gathers. |
| 3     | open   | `posdec` decomposition on the picked arrivals; validate forced/measure-dependent split against the 15/9-F-1A sonic log; held-out arrival calibration. |
| 4     | open   | (optional) Bodin-Sambridge rj-MCMC + simple PINNtomo as parallel comparators on the same picks. |

## Geometry

Known from the Volve documentation (DiscoverVolve / Equinor T&Cs):

- 151 surface shots, EW line, 100 m spacing, x in [3700, 18700] m,
  source 15 m below sea surface.
- 467 downhole 4-component receivers in 15/9-F-1A at MD 1000-7990 m,
  15 m spacing.
- 8 ms sampling, 2001 samples per trace = 16 s record length.
- Max 70 517 shot-receiver picks if every pair is used.

The well easting (`X_WELL_M_PLACEHOLDER`) is a placeholder until the
real SEG-Y headers are read; the survey origin and coordinate frame
will be confirmed on first read.

## Data placement

Drop the downloaded files in `volve/data/` (gitignored):

```
volve/data/
  walkaway/                 # SEG-Y shot gathers from Well_logs/08.VSP_VELOCITY/
    <name>.sgy
    ...
  logs/
    15_9-F-1A.LAS           # sonic + density + GR for the tie well
```

Then:

```bash
uv run python -m volve.smoke                 # confirms ingestion still works
uv run python -m volve.geometry              # plots the survey geometry
```

Loaders from a Python session:

```python
from volve.load_vsp  import load_file, summarize_block, bin_to_shot_recv_tensor
from volve.load_logs import load_well_log, summary, plot_log

blk = load_file("volve/data/walkaway/<file>.sgy")
print(summarize_block(blk))                  # header sanity-check
arr, hdr = bin_to_shot_recv_tensor(blk, n_components=4)
# arr shape: (n_shots, n_recv, n_comp, n_samples)

log = load_well_log("volve/data/logs/15_9-F-1A.LAS")
print(summary(log))                          # Vp range, depth range
plot_log(log, out_path="volve_sonic.png")
```

## Access (one-time)

1. Sign up at `data.equinor.com` (B2C account, email-verified).
2. Click through the data-use agreement.
3. The portal mints a per-user SAS URL to the Azure Blob container.
4. Use **Azure Storage Explorer** (GUI) or **AZCopy** (CLI) to selectively
   pull. Example:

   ```bash
   azcopy copy "<SAS-URL>/Well_logs/08.VSP_VELOCITY/*" \
       ./volve/data/walkaway/ --recursive
   azcopy copy "<SAS-URL>/Well_logs/15_9-F-1A.LAS" \
       ./volve/data/logs/
   ```

5. Walkaway envelope ~2-5 GB; LAS bundle well under 100 MB.

## License

Equinor's license is more permissive than commonly cited - permits
commercial use of Adapted Material, requires attribution and
ShareAlike-style propagation of terms to derived data. Derived products
(picks, inverted models, figures) can be published with attribution.

## Conventions

See module docstrings for the full convention block. Briefly:

- All distances in metres; all times in seconds.
- z positive downward; the well sits at x = X_WELL_M (refined after read).
- LAS DEPT in metres; DT in us/ft converted to Vp(km/s) = 304.8 / DT.

## What we still don't know until data arrives

- Exact SEG-Y trace ordering (shot-receiver-component, or
  receiver-component-shot, or another). `bin_to_shot_recv_tensor` assumes
  the standard layout but ERRORS rather than silently reshapes; first
  real-file read will tell us.
- The CoordinateScalar in the headers. We apply it per SEG-Y rev 1 but
  some files use a different sign convention.
- Whether the 4 components are interleaved per receiver or stored as
  blocks. The smoke uses interleaved; we'll re-fold after the first
  header peek if needed.
