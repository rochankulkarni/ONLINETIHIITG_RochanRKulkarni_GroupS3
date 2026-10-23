# Day 9 : Planning How to Split the Data

## What We Found So Far

- No missing values, no duplicate rows, data is clean.
- 242 rows are label 0, 157 are label 1 overall, a bit uneven but not extreme.
- Each water condition has a decent mix of 0s and 1s.
- Big finding: the old threshold method does great at low turbidity (0% mistakes at 1-2 NTU) but falls apart at higher turbidity (50% mistakes at 4-5 NTU).

## How We'll Split the Data

- **Split ratio:** 70% train, 15% validation, 15% test.
- **Keep it fair:** we'll make sure the split keeps a similar mix of 0s and 1s in each part (split that keeps a similar mix of 0s and 1s), and also try to keep all 5 water conditions represented across train, validation, and test, not just stuffed into one part.
- **Small dataset warning:** since we only have about 80 rows per water condition, a 70/15/15 split means some parts will only have around 12 rows per condition. We'll keep this in mind when judging how solid our results are later.

## Imbalance and Outliers

- **Imbalance:** 242 vs 157 isn't extreme, so we won't use any special balancing tricks for now. We'll watch precision and recall closely instead of just trusting accuracy.
- **Outliers:** we're not removing any values. Every row is a real sensor reading under real water conditions, nothing looks like a mistake.

## What We Decided

1. Keep all 399 rows, no removals.
2. Don't use the `binary bit R` column for training, only `Binary bit S` as the real answer.
3. Split 70/15/15, keeping balance between the two labels and water conditions as even as possible across the splits.
4. No special balancing tricks for now, might revisit if a model performs badly on one class.
