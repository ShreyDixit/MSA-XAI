# Spatial agreement: numerical sensitivity

Repeated unlesioned inference differed by at most 1.19e-07 in an individual raw output value. Exact zero-change detection can therefore miss functionally unchanged filters when cuDNN kernels differ at floating-point precision. The primary analysis retains all positive changes as specified. This supplementary analysis excludes targets with mean absolute image change below 10⁻⁶ or 10⁻⁵, in raw [0,1] output units. These thresholds are conservative numerical checks; they do not establish a biological or perceptual relevance threshold.

Each scenario averages retained filters within images and bootstraps images with 10,000 resamples. The original spatial masks and null controls are unchanged; a small-change filter's attribution map can still serve as a same-layer mismatch reference. Intervals are paired, conditional on the saved estimates and checkpoint.

| Minimum mean change | Layer | Retained/total cases | Gain vs shuffle (95% CI) | Gain vs other filters (95% CI) |
|---:|---:|---:|---:|---:|
| 0 | 0 | 32768/32768 | 0.4879 [0.4788, 0.4966] | 0.4422 [0.4336, 0.4506] |
| 0 | 1 | 16384/16384 | 0.5212 [0.5113, 0.5315] | 0.4641 [0.4565, 0.4717] |
| 0 | 2 | 8192/8192 | 0.5074 [0.4991, 0.5162] | 0.4460 [0.4414, 0.4507] |
| 0 | 3 | 4096/4096 | 0.5134 [0.5069, 0.5200] | 0.4706 [0.4665, 0.4745] |
| 0 | 4 | 2048/2048 | 0.2256 [0.2185, 0.2329] | 0.2084 [0.2028, 0.2138] |
| 1e-06 | 0 | 30161/32768 | 0.5291 [0.5175, 0.5401] | 0.4809 [0.4660, 0.4963] |
| 1e-06 | 1 | 16278/16384 | 0.5245 [0.5152, 0.5345] | 0.4670 [0.4602, 0.4739] |
| 1e-06 | 2 | 8188/8192 | 0.5076 [0.4991, 0.5166] | 0.4462 [0.4417, 0.4509] |
| 1e-06 | 3 | 4095/4096 | 0.5134 [0.5068, 0.5200] | 0.4706 [0.4665, 0.4745] |
| 1e-06 | 4 | 2048/2048 | 0.2256 [0.2185, 0.2330] | 0.2084 [0.2029, 0.2138] |
| 1e-05 | 0 | 30072/32768 | 0.5289 [0.5172, 0.5399] | 0.4807 [0.4655, 0.4955] |
| 1e-05 | 1 | 16264/16384 | 0.5243 [0.5148, 0.5341] | 0.4668 [0.4598, 0.4738] |
| 1e-05 | 2 | 8185/8192 | 0.5075 [0.4990, 0.5167] | 0.4461 [0.4415, 0.4507] |
| 1e-05 | 3 | 4093/4096 | 0.5132 [0.5067, 0.5199] | 0.4705 [0.4663, 0.4743] |
| 1e-05 | 4 | 2048/2048 | 0.2256 [0.2184, 0.2331] | 0.2084 [0.2028, 0.2138] |

Positive lower interval bounds indicate image-level mean agreement above the control under that inclusion rule. Compare rows within each layer to assess whether primary conclusions depend on numerically tiny effects. This sensitivity check does not alter the primary dataset or regenerate Shapley Modes.
