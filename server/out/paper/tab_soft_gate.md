| split | system | n | AUC | Spearman | dAUC_vs_OR | dAUC_vs_OR_ci | dSp_vs_OR | dSp_vs_OR_ci | dAUC_vs_ORn16 | dAUC_vs_ORn16_ci | dSp_vs_ORn16 | dSp_vs_ORn16_ci |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | OpenReviewer-8B | 180 | 0.676 | 0.342 | nan | nan | nan | nan | nan | nan | nan | nan |
| dev | DeepReviewer-7B (fast) | 174 | 0.663 | 0.311 | -0.007 | [-0.106,+0.096] | -0.026 | [-0.209,+0.168] | nan | nan | nan | nan |
| dev | AI Scientist reviewer (Qwen3-32B) | 179 | 0.613 | 0.219 | -0.067 | [-0.173,+0.038] | -0.133 | [-0.328,+0.057] | nan | nan | nan | nan |
| dev | Qwen3-32B zero-shot | 178 | 0.589 | 0.208 | -0.083 | [-0.184,+0.016] | -0.126 | [-0.305,+0.054] | nan | nan | nan | nan |
| dev | OpenReviewer-8B n16 | 180 | 0.763 | 0.480 | 0.086 | [+0.012,+0.157] | 0.138 | [+0.008,+0.272] | nan | nan | nan | nan |
| dev | OpenReviewer-8B n16 + soft gate (ours) | 180 | 0.778 | 0.485 | 0.102 | [+0.024,+0.175] | 0.142 | [+0.011,+0.278] | 0.015 | [-0.006,+0.040] | 0.005 | [-0.036,+0.044] |
| test25 | OpenReviewer-8B | 200 | 0.729 | 0.450 | nan | nan | nan | nan | nan | nan | nan | nan |
| test25 | DeepReviewer-7B (fast) | 189 | 0.663 | 0.337 | -0.077 | [-0.155,-0.001] | -0.137 | [-0.261,-0.004] | nan | nan | nan | nan |
| test25 | AI Scientist reviewer (Qwen3-32B) | 200 | 0.619 | 0.252 | -0.109 | [-0.191,-0.027] | -0.198 | [-0.337,-0.058] | nan | nan | nan | nan |
| test25 | Qwen3-32B zero-shot | 200 | 0.604 | 0.314 | -0.124 | [-0.206,-0.041] | -0.136 | [-0.280,+0.008] | nan | nan | nan | nan |
| test25 | OpenReviewer-8B n16 | 200 | 0.770 | 0.545 | 0.042 | [-0.015,+0.105] | 0.095 | [-0.005,+0.198] | nan | nan | nan | nan |
| test25 | OpenReviewer-8B n16 + soft gate (ours) | 200 | 0.770 | 0.533 | 0.041 | [-0.016,+0.107] | 0.083 | [-0.018,+0.188] | -0.000 | [-0.021,+0.022] | -0.012 | [-0.046,+0.021] |
| test24 | OpenReviewer-8B | 200 | 0.652 | 0.332 | nan | nan | nan | nan | nan | nan | nan | nan |
| test24 | DeepReviewer-7B (fast) | 192 | 0.616 | 0.232 | -0.035 | [-0.127,+0.061] | -0.105 | [-0.261,+0.047] | nan | nan | nan | nan |
| test24 | AI Scientist reviewer (Qwen3-32B) | 200 | 0.638 | 0.222 | -0.013 | [-0.111,+0.090] | -0.110 | [-0.259,+0.041] | nan | nan | nan | nan |
| test24 | Qwen3-32B zero-shot | 200 | 0.638 | 0.255 | -0.014 | [-0.095,+0.065] | -0.077 | [-0.205,+0.055] | nan | nan | nan | nan |
| test24 | OpenReviewer-8B n16 | 200 | 0.697 | 0.401 | 0.045 | [-0.013,+0.107] | 0.069 | [-0.032,+0.176] | nan | nan | nan | nan |
| test24 | OpenReviewer-8B n16 + soft gate (ours) | 200 | 0.691 | 0.379 | 0.039 | [-0.023,+0.104] | 0.047 | [-0.054,+0.156] | -0.006 | [-0.022,+0.009] | -0.022 | [-0.051,+0.001] |
| pooled_test | OpenReviewer-8B | 400 | 0.690 | 0.398 | nan | nan | nan | nan | nan | nan | nan | nan |
| pooled_test | DeepReviewer-7B (fast) | 381 | 0.639 | 0.288 | -0.056 | [-0.118,+0.004] | -0.125 | [-0.226,-0.022] | nan | nan | nan | nan |
| pooled_test | AI Scientist reviewer (Qwen3-32B) | 400 | 0.629 | 0.236 | -0.061 | [-0.123,+0.005] | -0.162 | [-0.269,-0.052] | nan | nan | nan | nan |
| pooled_test | Qwen3-32B zero-shot | 400 | 0.621 | 0.287 | -0.070 | [-0.126,-0.010] | -0.111 | [-0.212,-0.009] | nan | nan | nan | nan |
| pooled_test | OpenReviewer-8B n16 | 400 | 0.735 | 0.480 | 0.045 | [+0.003,+0.087] | 0.082 | [+0.010,+0.157] | nan | nan | nan | nan |
| pooled_test | OpenReviewer-8B n16 + soft gate (ours) | 400 | 0.732 | 0.465 | 0.041 | [-0.002,+0.085] | 0.067 | [-0.007,+0.143] | -0.003 | [-0.016,+0.010] | -0.015 | [-0.036,+0.006] |
