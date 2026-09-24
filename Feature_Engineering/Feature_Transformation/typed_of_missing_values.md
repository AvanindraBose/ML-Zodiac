# Types of Missing Values

There are 3 types of missingness. The difference between them is **WHY** the data is missing.

---

There are 2 types of data : 

1. Observed Data -> Which is present in the Data.

2. Unobserved Data -> Which is missing from the data

---

## 1. MCAR — Missing Completely At Random

**Simple meaning:** The data is missing for a totally random reason — it has **nothing to do** with any value (observed or unobserved).

**Test:** If you removed the missing rows, the remaining data would still look like a fair, random sample of the whole dataset.

**Example:**
A weighing scale in a hospital randomly malfunctions on some days and fails to record a few patients' weights. The malfunction has nothing to do with the patient's actual weight, age, or any other factor — it's pure bad luck/machine error.

✅ **Safest type** — missingness doesn't bias your analysis.

---

## 2. MAR — Missing At Random

**Simple meaning:** The data is missing for a reason, but that reason is **explained by observed data that is present in your dataset**.

**Intuition** : The missingness in a column can be explained using information that is observed.

        It can come from:

                1. the same column's observed values, or

                2. a different observed column.

**Test:** If you know the value of some **other observed column**, you can explain/predict why this value is missing.

**Example:**
In a survey, **men** are less likely to answer a question about "emotional wellbeing" than women. The missingness in "emotional wellbeing" isn't random — but it's fully explained by the **Gender** column, which you *do* have in your data.

✅ **Fixable** — since the reason is in your data, you can use techniques like multiple imputation (using Gender to help predict/fill the missing values).

---

## 3. MNAR — Missing Not At Random

**Simple meaning:** The data is missing because of the **unobserved data**, or because of **some other factor you never measured/recorded at all**.

**Intuition** : The missingness in a column depends on information that is unobserved.

        It can be:

                1. the unobserved value of the same column, or

                2. an unobserved value from a different column.

**Test:** Even if you look at all your observed columns, you *still* can't explain why the value is missing.

**Example A (missing because of its own value):**
People with **very high income** are less likely to disclose their income in a survey. The missingness in "Income" depends on Income itself — which you don't know, since it's missing.

**Example B (missing because of an unmeasured factor):**
People with high **anxiety about judgment** skip the income question — but you never asked about "anxiety" anywhere in your survey. The true reason is invisible to you because that column doesn't exist in your data.

⚠️ **Hardest to fix** — the reason is hidden, so even smart imputation methods can't fully correct for it. Usually requires **sensitivity analysis** (testing multiple assumptions to see how much your results change) or collecting more data.

---

## Quick Comparison Table

| Type | Depends on... | Example | Fixable? |
|------|---------------|---------|----------|
| **MCAR** | Nothing (pure random) | Machine randomly fails to record weight | ✅ Easiest |
| **MAR** | An observed column | Men skip a question more than women | ✅ Fixable using other columns |
| **MNAR** | The missing value itself, or an unmeasured column | High earners don't report income | ⚠️ Hardest — needs assumptions |

---

## One-Line Recall Trick

- **MCAR** → Random glitch, no reason at all.
- **MAR** → There's a reason, and it's *in* your data.
- **MNAR** → There's a reason, but it's *hidden* (either the value itself or a column you never collected).