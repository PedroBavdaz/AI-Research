import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

# Given data
data = {
    'Control': [80.369, 87.446, 90.015, 99.046, 75.477, 87.231, 91.754, 87.785, 77.8, 62.646, 84.738, 84.877, 73.277, 84.523, 70.877],
    'Friend': [99.692, 83.4, 102.154, 80.277, 88.015, 92.492, 91.354, 100.877, 101.062, 81.6, 89.815, 98.2, 76.908, 86.985, 97.046],
    'Pet': [69.169, 70.169, 75.985, 86.446, 68.862, 64.169, 97.538, 85, 72.262, 58.692, 79.662, 69.231, 69.538, 70.077, 65.446]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# a. Side-by-Side Box Plot
df.boxplot(grid=False)
plt.title('Heart Rates by Group')
plt.ylabel('Heart Rate (beats per minute)')
plt.show()

# b. Null Hypothesis
# Already stated above

# c. ANOVA Test
f_stat, p_value = f_oneway(df['Control'], df['Friend'], df['Pet'])
print(f"ANOVA F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# d. Bonferroni Pairwise Test
if p_value < 0.05:
    pairwise_results = pairwise_tukeyhsd(
        df.values.flatten(), df.columns.repeat(len(df)))
    print(pairwise_results)
