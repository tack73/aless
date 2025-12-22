# 論文用：重回帰分析の記述

## Method（方法）

### Statistical Analysis（統計分析）

To examine the independent effects of childhood environmental factors on current insect dislike scores, we conducted a multiple regression analysis using ordinary least squares (OLS) estimation. The dependent variable was the Insect Dislike Score, computed as the sum of responses to 11 items (range: 11-66), with higher scores indicating greater dislike of insects.

Independent variables included four childhood environmental factors: (1) frequency of outdoor play in natural environments (mountains, rivers, seas, rice fields), (2) general reading habits, (3) frequency of reading insect-related books, and (4) residential area type. Since these variables were measured on ordinal scales with three levels (rarely, sometimes, frequently for the first three factors; rural to urban for residential area), we converted them into dummy variables to avoid assuming equal intervals between categories.

For each ordinal variable, we created dummy variables with the lowest category as the reference group (rarely for behavioral frequencies, rural for residential area). Specifically:
- Nature Contact: two dummy variables (Sometimes, Frequent; reference: Rarely)
- Reading Habit: two dummy variables (Sometimes, Frequent; reference: Rarely)  
- Insect Book Reading: two dummy variables (Sometimes, Frequent; reference: Rarely)
- Residential Area: three dummy variables (Regional City, Suburban, Urban; reference: Rural)

This approach allows for non-linear relationships between ordinal categories and does not assume equal distances between levels. The regression coefficients represent the difference in insect dislike scores compared to the reference category, holding all other variables constant. All analyses were conducted using Python 3.x with the statsmodels library (version X.X.X).

---

## Results（結果）

### Multiple Regression Analysis

The multiple regression model examining the effects of childhood environmental factors on insect dislike scores was statistically significant (F(9, 40) = 3.42, p = 0.003), explaining 43.5% of the variance in insect dislike scores (R² = 0.435, adjusted R² = 0.308).

**Insect Book Reading Frequency** showed the strongest effect on reducing insect dislike. Participants who frequently read insect-related books in childhood had significantly lower dislike scores (β = -19.33, SE = 6.44, p = 0.005) compared to those who rarely read such books, representing approximately a 19-point decrease. Even occasional reading of insect books showed a significant effect (β = -8.77, SE = 4.19, p = 0.043).

**General Reading Habits** also demonstrated significant effects. Both frequent readers (β = -12.76, SE = 5.73, p = 0.032) and occasional readers (β = -13.68, SE = 5.68, p = 0.021) showed significantly lower insect dislike scores compared to those who rarely read, with approximately 13-point reductions.

**Nature Contact Frequency** showed minimal and non-significant effects. Neither occasional outdoor play in natural environments (β = -0.06, SE = 4.04, p = 0.989) nor frequent play (β = -0.16, SE = 4.77, p = 0.974) significantly predicted insect dislike scores when controlling for other factors.

**Residential Area** showed no significant effects on insect dislike. Compared to rural areas, growing up in regional cities (β = 9.63, SE = 12.76, p = 0.455), suburban areas (β = 14.78, SE = 12.04, p = 0.227), or urban areas (β = 7.97, SE = 12.08, p = 0.513) did not significantly affect current insect dislike levels.

Table 1 presents the complete regression results with coefficients, standard errors, and significance levels for all variables.

---

## Table 1. Multiple Regression Analysis Results

| Variable                  | Coefficient | SE    | p-value | Significance |
| ------------------------- | ----------- | ----- | ------- | ------------ |
| **Nature Contact**        |             |       |         |              |
| Sometimes (vs. Rarely)    | -0.06       | 4.04  | 0.989   | n.s.         |
| Frequent (vs. Rarely)     | -0.16       | 4.77  | 0.974   | n.s.         |
| **Reading Habit**         |             |       |         |              |
| Sometimes (vs. Rarely)    | -13.68      | 5.68  | 0.021   | *            |
| Frequent (vs. Rarely)     | -12.76      | 5.73  | 0.032   | *            |
| **Insect Book Reading**   |             |       |         |              |
| Sometimes (vs. Rarely)    | -8.77       | 4.19  | 0.043   | *            |
| Frequent (vs. Rarely)     | -19.33      | 6.44  | 0.005   | **           |
| **Residential Area**      |             |       |         |              |
| Regional City (vs. Rural) | 9.63        | 12.76 | 0.455   | n.s.         |
| Suburban (vs. Rural)      | 14.78       | 12.04 | 0.227   | n.s.         |
| Urban (vs. Rural)         | 7.97        | 12.08 | 0.513   | n.s.         |

*Note.* N = 50. R² = 0.435, adjusted R² = 0.308, F(9, 40) = 3.42, p = 0.003.  
*p < .05. **p < .01. ***p < .001. n.s. = not significant.

---

## Discussion Points（考察のポイント）

1. **Reading insect-related books showed the strongest independent effect** on reducing insect dislike, even after controlling for general reading habits and nature exposure.

2. **General reading habits also significantly reduced insect dislike**, suggesting that intellectual curiosity and learning may reduce fear of the unknown.

3. **Direct nature contact had no significant independent effect** when controlling for other factors, suggesting that knowledge acquisition (through reading) may be more important than mere exposure.

4. **Urban-rural differences were not significant**, contrary to common assumptions that urban upbringing increases insect aversion.

5. The findings suggest that **educational interventions using books and written materials** may be more effective than simple nature exposure programs for reducing insect phobia.
