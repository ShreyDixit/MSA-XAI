# DCGAN quantification: larger CelebA generator

## Dataset and compatibility

One historical dataset: 63584 signed, unnormalized filter–image modes, 32 fixed latent vectors, 1,984 hidden filters (layers 0–4: 1,024/512/256/128/64), and three final RGB players (layer 5). Each mode is 3×128×128. The historical Shapley game included both hidden and output units; output-channel gating is therefore part of the coalition context for the hidden modes. No new Shapley estimates were mixed in.

Checkpoint `/home/shrey/MSA-XAI/DCGAN/saved_results/celebA_GAN_bigger_smodel.pth`; SHA-256 `11ee12592097d9ecf222b8c17dcc93bda2271799c4e2ba6cf96580c28e25c555`. Full identities, versions, precision settings, and perturbation definitions are in `manifest.json`; ordered unit mapping and all latent values are in CSVs.

Summed modes reconstruct the unlesioned checkpoint output minus the fully lesioned output with maximum absolute residual 1.19e-07 and mean absolute residual 2.41e-09. All 32 images match their own latent indices; fully lesioned output is zero. Repeated unlesioned inference differs by at most 1.19e-07. This supports compatibility; the historical run did not store input hashes. Inference uses the notebook's TF32/cuDNN defaults rather than forcing a different deterministic algorithm.

## Option 2: spatial intervention agreement

For each hidden filter and image, A(x,y)=mean_RGB(|signed Shapley mode|) and D(x,y)=mean_RGB(|unlesioned image − single-filter-lesioned image|). Lesions zero one post-BatchNorm/ReLU channel using `Generator.forward` in evaluation mode, with the same checkpoint and fixed latents as the modes. Raw outputs are cached separately by layer.

The selected mask contains exactly ceil(0.20×128×128)=3,277 pixels (20.00122070%); boundary ties are broken by seeded random ordering. Capture is sum(D inside mask)/sum(D). Zero-change cases are undefined and excluded, with counts reported separately. Zero-attribution cases are also excluded; near-zero change (positive mean change <10⁻⁶) is flagged but retained in the primary analysis.

Controls: 100 seeded uniform spatial permutations of each attribution mask; their exact expected capture is the selected pixel fraction. The mismatched control is the exact mean capture over every other eligible filter's attribution mask in the same layer and image. It is computed from summed masks, without sampling filter pairs. Uniform permutation destroys spatial structure; the layer control retains layer-specific spatial biases.

Scores are averaged equally across valid filters within each image/layer, then equally across images. Brackets below are 95% percentile intervals from 10,000 image bootstrap resamples; gain intervals use paired differences. They quantify variation over these saved images, conditional on one checkpoint and the saved Shapley estimates. Pixels and filters are not treated as independent replicates. No per-filter significance claims are made.

| Hidden layer | Matched capture | Gain vs spatial shuffle | Gain vs other filters |
|---|---:|---:|---:|
| 0 | 0.6879 [0.6787, 0.6968] | 0.4879 [0.4786, 0.4968] | 0.4422 [0.4334, 0.4505] |
| 1 | 0.7212 [0.7110, 0.7315] | 0.5212 [0.5110, 0.5318] | 0.4641 [0.4563, 0.4716] |
| 2 | 0.7074 [0.6989, 0.7162] | 0.5074 [0.4991, 0.5164] | 0.4460 [0.4415, 0.4507] |
| 3 | 0.7134 [0.7069, 0.7202] | 0.5134 [0.5068, 0.5200] | 0.4706 [0.4666, 0.4745] |
| 4 | 0.4256 [0.4183, 0.4329] | 0.2256 [0.2184, 0.2329] | 0.2084 [0.2029, 0.2137] |

Of 63,488 hidden filter–image cases, 0 have zero change, 0 have zero attribution, and 2,718 have positive mean change below 10⁻⁶. See `agreement_exclusions.csv` for layer counts.

Agreement describes localization of single-filter effects. Shapley Modes average effects over coalitions, so perfect agreement with removal from the fully intact network is not required. Exceeding spatial shuffle alone can reflect generic spatial organization; the same-layer mismatched comparison addresses filter specificity.

## Option 3: layer-wise spatial scale

Each signed RGB channel is demeaned over space before a full two-dimensional FFT. Power is summed across channels (no cancellation across RGB), with DC explicitly removed. The centroid weights radial frequency sqrt(fx²+fy²), in cycles/pixel, by this combined power and divides by sqrt(0.5²+0.5²), the 2-D Nyquist corner. The normalized range is 0–1, where larger values indicate finer spatial variation. Zero non-DC-energy maps receive NaN and are excluded from scale summaries. Modes are not rectified or display-normalized before this calculation.

Primary estimates use no taper. A separable Hann taper applied after demeaning provides a sensitivity check for image-edge discontinuities; its DC is also discarded. Distributions show individual filter–image values. Layer means and uncertainty first average valid filters per image, then bootstrap images as above.

| Hidden layer | Spectral centroid | Hann sensitivity centroid |
|---|---:|---:|
| 0 | 0.1176 [0.1096, 0.1272] | 0.1163 [0.1080, 0.1261] |
| 1 | 0.1097 [0.1075, 0.1120] | 0.1061 [0.1041, 0.1082] |
| 2 | 0.1640 [0.1608, 0.1671] | 0.1611 [0.1581, 0.1638] |
| 3 | 0.2558 [0.2535, 0.2582] | 0.2514 [0.2479, 0.2549] |
| 4 | 0.5483 [0.5417, 0.5552] | 0.5552 [0.5484, 0.5624] |

Layer means do not increase strictly with depth. Within-image last-minus-first centroid: 0.4307 [0.4239, 0.4376]. 15/32 images have strictly increasing layer means. Interpret this as spatial-scale evidence, not evidence for particular facial semantics.

Zero-energy modes: 0 hidden and 0 output cases. Final RGB players are reported separately in `spectral_output_rgb_summary.csv` and `figures/output_rgb_spatial_scale.*`; they are excluded from the hidden-layer trend.

## Artifacts and limitations

- `agreement_by_filter_image.csv`, `agreement_by_image_layer.csv`, and `agreement_by_layer.csv`: spatial results and paired gains.
- `spectral_by_filter_image.csv`, `spectral_by_image_layer.csv`, `spectral_by_layer.csv`, and `spectral_trend_*.csv`: signed-map scale and image-level trends.
- `figures/`: PDF/SVG/600-dpi PNG comparisons in the MLP/LLM visual style, layer distributions, separate RGB results, and illustrative examples from notebook image 30 (filters 72/50/7/12/8). Display heatmaps are scaled to their own maxima for qualitative localization only. Final figure layers 1–5 correspond to CSV layers 0–4. Example selection is documented in `qualitative_examples.json`; presentation provenance is in `figure_manifest.json`.
- `../cache/`: raw single-filter outputs, completion flags, baseline, copied latents, summed modes, and packed attribution masks. Input fingerprints guard against incompatible cache reuse.

The 32 saved images are one fixed sample, and only one trained checkpoint is examined. No uncertainty over training seeds or historical Shapley permutations can be recovered from averaged modes. Frequency measurements depend on power weighting, image boundaries, and output resolution; the Hann check probes one boundary choice. Spatial masks fix the requested 20% threshold rather than selecting it based on the outcome. No manual semantic annotation or quantitative semantic claim is used.
