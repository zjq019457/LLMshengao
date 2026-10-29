| split | system | n | AUC | Spearman | dAUC_vs_base | dAUC_vs_base_ci | dSp_vs_base | dSp_vs_base_ci | dAUC_vs_best_single | dAUC_vs_best_single_ci | dSp_vs_best_single | dSp_vs_best_single_ci | dAUC_vs_ensemble | dAUC_vs_ensemble_ci | dSp_vs_ensemble | dSp_vs_ensemble_ci | best_single |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test25 | deepreviewer-7b:fast | 200 | 0.656 | 0.320 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test25 | deepreviewer-7b:fast+fusion (M3a) | 200 | 0.648 | 0.299 | -0.008 | [-0.055,+0.039] | -0.022 | [-0.096,+0.055] | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test25 | qwen3-32b-awq:s10 | 200 | 0.604 | 0.314 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test25 | qwen3-32b-awq:s10+fusion (M3a) | 200 | 0.638 | 0.267 | 0.034 | [-0.041,+0.110] | -0.047 | [-0.175,+0.078] | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test25 | qwen3-32b-awq:ais | 200 | 0.619 | 0.252 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test25 | qwen3-32b-awq:ais+fusion (M3a) | 200 | 0.626 | 0.209 | 0.007 | [-0.065,+0.078] | -0.043 | [-0.167,+0.074] | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test25 | openreviewer-8b | 200 | 0.729 | 0.450 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test25 | openreviewer-8b+fusion (M3a) | 200 | 0.711 | 0.370 | -0.017 | [-0.063,+0.026] | -0.080 | [-0.165,+0.001] | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test25 | ensemble (no audit) | 200 | 0.717 | 0.440 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test25 | ensemble+fusion (M3b) | 200 | 0.709 | 0.403 | nan | nan | nan | nan | -0.019 | [-0.067,+0.029] | -0.047 | [-0.136,+0.044] | -0.008 | [-0.038,+0.022] | -0.036 | [-0.084,+0.010] | openreviewer-8b |
| test24 | deepreviewer-7b:fast | 200 | 0.605 | 0.213 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test24 | deepreviewer-7b:fast+fusion (M3a) | 200 | 0.600 | 0.167 | -0.005 | [-0.060,+0.050] | -0.046 | [-0.140,+0.043] | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test24 | qwen3-32b-awq:s10 | 200 | 0.638 | 0.255 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test24 | qwen3-32b-awq:s10+fusion (M3a) | 200 | 0.577 | 0.152 | -0.060 | [-0.128,+0.009] | -0.103 | [-0.225,+0.013] | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test24 | qwen3-32b-awq:ais | 200 | 0.638 | 0.222 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test24 | qwen3-32b-awq:ais+fusion (M3a) | 200 | 0.569 | 0.116 | -0.070 | [-0.146,+0.005] | -0.106 | [-0.227,+0.013] | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test24 | openreviewer-8b | 200 | 0.652 | 0.332 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test24 | openreviewer-8b+fusion (M3a) | 200 | 0.628 | 0.248 | -0.024 | [-0.070,+0.025] | -0.084 | [-0.159,-0.011] | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test24 | ensemble (no audit) | 200 | 0.673 | 0.337 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| test24 | ensemble+fusion (M3b) | 200 | 0.661 | 0.289 | nan | nan | nan | nan | 0.009 | [-0.044,+0.064] | -0.043 | [-0.132,+0.041] | -0.012 | [-0.045,+0.026] | -0.048 | [-0.101,+0.005] | openreviewer-8b |
| dev | deepreviewer-7b:fast | 180 | 0.667 | 0.311 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| dev | deepreviewer-7b:fast+fusion (M3a) | 180 | 0.686 | 0.357 | 0.019 | [-0.030,+0.070] | 0.046 | [-0.035,+0.127] | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| dev | qwen3-32b-awq:s10 | 180 | 0.586 | 0.201 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| dev | qwen3-32b-awq:s10+fusion (M3a) | 180 | 0.631 | 0.327 | 0.045 | [-0.030,+0.120] | 0.125 | [+0.002,+0.247] | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| dev | qwen3-32b-awq:ais | 180 | 0.612 | 0.214 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| dev | qwen3-32b-awq:ais+fusion (M3a) | 180 | 0.632 | 0.296 | 0.020 | [-0.047,+0.091] | 0.082 | [-0.036,+0.205] | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| dev | openreviewer-8b | 180 | 0.676 | 0.342 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| dev | openreviewer-8b+fusion (M3a) | 180 | 0.688 | 0.391 | 0.012 | [-0.039,+0.066] | 0.048 | [-0.037,+0.141] | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| dev | ensemble (no audit) | 180 | 0.716 | 0.399 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| dev | ensemble+fusion (M3b) | 180 | 0.727 | 0.433 | nan | nan | nan | nan | 0.051 | [-0.015,+0.120] | 0.091 | [-0.027,+0.218] | 0.011 | [-0.020,+0.045] | 0.034 | [-0.020,+0.087] | openreviewer-8b |
