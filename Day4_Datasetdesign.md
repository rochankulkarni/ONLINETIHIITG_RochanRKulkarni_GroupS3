# Day 4 : Planning the Dataset

## Expected Files, Columns, Labels & Metadata (Before Using Final Dataset)

- **Expected columns:** received analog signal value, transmitted bit label (target), turbidity/NTU level (extra information/condition).
- **Expected extra information:** which turbidity condition each row corresponds to, ideally covering multiple NTU levels so performance across different water conditions can actually be tested.
- **Expected structure:** one row per transmission/measurement, consistently labelled.

## Likely Risks (Checklist)

| Risk Category | Specific Risk | Likelihood (pre-inspection estimate) |
|---|---|---|
| Missing data | Gaps in signal readings or missing NTU labels | Low-Medium |
| Annotation | Mislabeled bits due to manual/hardware logging errors | Low |
| Imbalance | Class imbalance (more 0s than 1s or vice versa) | Medium |
| Imbalance | Turbidity-condition imbalance (uneven rows per NTU level) | Medium |
| Leakage | A column that directly or indirectly reveals the target (e.g. a pre-computed prediction column) | Medium : worth checking explicitly |
| Sample size | Small overall dataset size, limiting split reliability | Medium-High : flagged as a risk to monitor once the real row count is known |
| Duplication | Repeated readings due to low sensor resolution, mistaken for data errors | Medium : depends on hardware ADC resolution |

## Train/Validation/Test Split : What Must Be Preserved

- **Class balance:** split that keeps a similar mix of 0s and 1s so the ratio of 0s to 1s is consistent across train/val/test.
- **NTU condition coverage:** each split should include data from every available turbidity condition, so validation/test performance reflects real-world variation across water conditions, not just one easy condition.
- **No using information that gives away the answer across splits:** ensure no reading is duplicated across both train and test in a way that inflates test performance.
- **Reproducibility:** fixed random seed for the split, documented and reused across all experiments.

## Dataset-Risk Checklist (Summary)

- [ ] Check for missing values in every column.
- [ ] Check for duplicate rows and understand their cause before deciding to keep/remove.
- [ ] Check balance between the two labels (bit label distribution), overall and per NTU level.
- [ ] Check row count per NTU condition : flag if any condition is too sparse for reliable testing.
- [ ] Check for any column that might leak the answer (e.g. a pre-existing prediction column).
- [ ] Confirm all labels are valid (only 0/1, no corrupted entries).
- [ ] Confirm feature values fall within an expected, physically plausible range.
