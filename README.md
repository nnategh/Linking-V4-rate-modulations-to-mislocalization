MATLAB and Python code for:

> **Limits of V4 perisaccadic firing rate modulations in explaining perceptual mislocalization**

## Structure
- `scripts/` — scripts that load data and generate figures
- `functions/` — analysis functions
- `data/` — location for `.mat` files
- `decoder/` — # Python: Neural decoding workflow (PCA-GPR)

## Figure mapping

| Script | Figure |
|---|---|
| `plot_misloc_locerr.m` | Fig. 1C-D, Fig. 4A-B |
| `plot_modulation_index.m` | Fig. 2, Fig. 4C-D |
| `plot_center_of_mass.m` | Fig. 3B-C |

## Decoder files (`decoder/`)

- `GPR_param_search.py` — Stage 1: shared GPR alpha parameter search
- `GPR_cross_test.py` — Stage 2: cross-condition decoding & statistical testing
- `gpr_cross_pca_utils.py` — shared PCA, GPR, and cross-validation utilities
- `thor_ozzy_combined_utils.py` — session discovery and parameter lookup helpers

## Decoding workflow
### Step 1: Parameter Search
```bash
python decoder/GPR_param_search.py \
  --thor-base-data-dir "/path/to/thor_data" \
  --ozzy-base-data-dir "/path/to/ozzy_data"

### Step 2: Cross-Condition Analysis
python decoder/GPR_cross_test.py \
  --thor-base-data-dir "/path/to/thor_data" \
  --ozzy-base-data-dir "/path/to/ozzy_data"
