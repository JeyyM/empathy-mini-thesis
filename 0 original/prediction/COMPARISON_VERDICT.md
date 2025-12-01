# MODEL COMPARISON: DID PERSONALITY ACTUALLY IMPROVE PREDICTIONS?

## **ANSWER: NO ❌**

The personality-enhanced model **performed significantly WORSE** than the original fusion model.

---

## Performance Metrics Comparison

| Metric | Original (Fusion Only) | Enhanced (+ Personality) | Change |
|--------|------------------------|--------------------------|---------|
| **R²** | **0.593** (59.3%) | 0.347 (34.7%) | **-41.5% ⬇️** |
| **Pearson r** | **0.770***  | 0.589* | **-23.5% ⬇️** |
| **MAE** | **10.48%** | 13.03% | **+24.3% ⬆️** (worse) |
| **RMSE** | **12.35%** | 17.11% | **+38.6% ⬆️** (worse) |
| **p-value** | 0.0008 (***) | 0.021 (*) | Less significant |

---

## What Went Wrong?

### 1. **Worse Prediction Accuracy**
- Original model: **10.48% average error**
- Enhanced model: **13.03% average error** (+24% worse)
- The model makes less accurate predictions with personality added

### 2. **Lower Explanatory Power**
- Original R²: **59.3%** of variance explained
- Enhanced R²: **34.7%** of variance explained
- Lost ability to explain **24.6%** of variance

### 3. **Mixed Individual Results**
- **7/15 improved** (47%)
- **8/15 worsened** (53%)
- Not statistically significant (p = 0.46)

### 4. **Group-Level Inconsistency**

| Group | Original MAE | Enhanced MAE | Change |
|-------|--------------|--------------|--------|
| **Similar** | 13.04% | **7.53%** | ✅ **-42%** (improved) |
| **Opposing** | 8.73% | 12.17% | ❌ +39% (worsened) |
| **Neutral** | 9.67% | 19.39% | ❌ **+100%** (doubled!) |

Only the Similar group improved; Neutral and Opposing groups got much worse.

---

## Why Did This Happen?

### Possible Explanations:

1. **Different Feature Sets**
   - Original model: Top 15 features selected from 273 available
   - Enhanced model: Different subset of 15 features + 8 personality
   - Not apples-to-apples comparison

2. **Missing Data**
   - 1 participant (EthanPlaza) missing personality data
   - This introduces bias in the enhanced model

3. **Overfitting**
   - Adding 8 more features to n=14 sample
   - Ratio of features to samples became worse (23:14 vs 15:15)
   - Model may be learning noise instead of signal

4. **Personality Not Predictive (for this task)**
   - Self-reported traits may not capture relevant aspects
   - Communication style ≠ emotional processing ability
   - The 8 traits measured might not be the right ones

5. **Feature Selection Issues**
   - The original top 15 features were optimized for prediction
   - Adding personality disrupted the optimal feature combination

---

## Individual-Level Analysis

### Biggest Improvements:
1. **KeithziCantona** (Similar): 21.6% → 0.4% error (✅ **-21.2%**)
2. **MaggieOng** (Similar): 16.0% → 5.8% error (✅ **-10.2%**)
3. **EthanOng** (Similar): 16.6% → 7.9% error (✅ **-8.7%**)

### Biggest Regressions:
1. **MiguelNg** (Neutral): 0.7% → 29.1% error (❌ **+28.4%**)
2. **MarwahMuti** (Opposing): 12.3% → 28.7% error (❌ **+16.4%**)
3. **RandellFabico** (Neutral): 20.1% → 36.1% error (❌ **+16.1%**)

---

## Statistical Tests

### Paired t-test:
- t-statistic: -0.755
- p-value: **0.463**
- Result: **NOT significant** at α=0.05

### Wilcoxon signed-rank test:
- p-value: **0.561**
- Result: **NOT significant** at α=0.05

**Conclusion**: The differences are not statistically significant. Adding personality neither significantly improved nor worsened predictions overall.

---

## Accuracy Distribution

|Category|Original|Enhanced|Change|
|--------|--------|--------|------|
|🟢 Excellent (<5%)|4/15 (27%)|5/15 (33%)|+1|
|🟡 Good (5-10%)|3/15 (20%)|3/15 (20%)|0|
|🟠 Fair (10-15%)|4/15 (27%)|2/15 (13%)|-2|
|🔴 Poor (>15%)|4/15 (27%)|5/15 (33%)|+1|

Slight shift toward extremes (more excellent, more poor).

---

## Key Takeaway: Why the Earlier Analysis Was Misleading

### The Initial Comparison Was Flawed:

**Earlier I compared:**
- Emotional features only (R² = 0.167)
- vs Combined (R² = 0.347)

This showed **+107% improvement**! ✅

**BUT** - the "emotional only" baseline was artificially poor because:
1. Used a **different subset** of 15 features
2. Not the **optimal** features for prediction
3. The original fusion model used the **best** 15 features (R² = 0.593)

### The Correct Comparison:
- **Original best model** (R² = 0.593)
- vs Enhanced model (R² = 0.347)
- Result: **-41.5% regression** ❌

---

## Recommendations

### ❌ Don't Use Personality Enhancement
The current implementation makes predictions **worse**, not better.

### ✅ Stick with Original Fusion Model
- R² = 0.593
- MAE = 10.48%
- r = 0.770*** (highly significant)

### 🔬 If You Want to Try Personality Again:
1. **Collect complete data** for all 15 participants
2. **Use validated personality measures** (Big Five, etc.)
3. **Test different features** - maybe these 8 traits aren't the right ones
4. **Use feature selection** to pick optimal emotional + personality combo
5. **Increase sample size** to avoid overfitting with more features
6. **Consider interactions** between personality and emotions

---

## Final Verdict

### Original Fusion Model Wins 🏆

**Original Model Performance:**
- Explains 59.3% of variance
- Average error: ±10.5%
- Highly significant: p < 0.001
- Consistent across all groups

**Enhanced Model Performance:**
- Explains only 34.7% of variance
- Average error: ±13.0%
- Barely significant: p = 0.021
- Inconsistent across groups

**Bottom Line:** Adding personality traits **decreased** model performance by reducing R² from 0.593 to 0.347 (-41.5%). The original fusion model is superior.

---

*Analysis Date: November 30, 2025*
