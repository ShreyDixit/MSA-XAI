#!/usr/bin/env python3
"""Build a short illustrated Markdown/PDF report from the saved quantitative results.

Uses only the standard library and an installed pdflatex executable. Does not run
inference or change analysis results. Rebuild: python DCGAN/quantification/build_illustrated_report.py
"""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STEM = "ILLUSTRATED_REPORT"


def read_csv(name):
    with (RESULTS / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    agreement = read_csv("agreement_by_layer.csv")
    spectral = read_csv("spectral_by_layer.csv")
    trends = {r["metric"]: r for r in read_csv("spectral_trend_summary.csv")}
    rgb = {r["channel"]: float(r["mean"]) for r in read_csv("spectral_output_rgb_summary.csv")}
    exclusions = read_csv("agreement_exclusions.csv")
    qc = json.loads((RESULTS / "compatibility.json").read_text())
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    trend_images = read_csv("spectral_trend_by_image.csv")
    increasing = sum(r["strictly_increasing"].lower() == "true" for r in trend_images)
    near_zero = sum(int(r["near_zero_change"]) for r in exclusions)
    cases = sum(int(r["maps"]) for r in exclusions)
    zero_change = sum(int(r["zero_change"]) for r in exclusions)
    zero_attr = sum(int(r["zero_attribution"]) for r in exclusions)
    zero_energy = sum(int(r["zero_energy_maps"]) for r in spectral if r["metric"] == "spectral_centroid")

    def value(rows, layer, metric):
        return float(next(r["mean"] for r in rows if int(r["layer"]) == layer and r["metric"] == metric))

    capture = [value(agreement, layer, "capture") for layer in range(5)]
    mismatch = [value(agreement, layer, "mismatched_mean") for layer in range(5)]
    scale = [value(spectral, layer, "spectral_centroid") for layer in range(5)]
    delta = trends["last_minus_first"]
    title = "DCGAN results: where filters matter and how their effects change"
    subtitle = "Plain-language findings with complete technical methods | Larger CelebA model"
    pages = [[] for _ in range(6)]

    def text(page, content):
        pages[page].append(("text", content))

    def heading(page, content):
        pages[page].append(("heading", content))

    def figure(page, name, caption, width=1., height=3.):
        pages[page].append(("figure", name, caption, width, height))

    def equation(page, expression):
        pages[page].append(("equation", expression))

    def table(page, headers, rows):
        pages[page].append(("table", headers, rows))

    def estimate(rows, layer, metric, factor=1, digits=3):
        row = next(r for r in rows if int(r["layer"]) == layer and r["metric"] == metric)
        mean, low, high = (float(row[k]) * factor for k in ("mean", "ci025", "ci975"))
        return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"

    heading(0, "1. Does a filter's map point to where the image actually changes?")
    text(0, "A filter is one internal channel in the generator. Its Shapley Mode is a map of "
         "its contribution to the output image, averaged over many combinations of active filters. "
         "Here we ask whether that map helps predict where the image changes when we switch off "
         "that filter on its own.")
    text(0, "How the test works. Mark the 20% of image pixels with the strongest attribution. "
         "Remove the filter, measure how much every pixel changes, and ask how much of the total "
         "change falls in the marked area. A score of 70% means that 70% of the change magnitude "
         "is concentrated in just 20% of the image. It does not mean that 70% of pixels changed.")
    figure(0, "spatial_agreement",
           "Figure 1. Higher means better localization. Navy: the correct filter's map. Purple: "
           "other filters' maps from the same layer. Grey: randomly shuffled locations, which "
           "capture about 20% by chance. Small bars show uncertainty across the 32 images. "
           "Layers are numbered 1-5 from earliest to latest throughout these figures.", 1., 3.25)
    text(0, f"What we found. The correct maps capture {min(capture):.1%}-{max(capture):.1%} "
         "of the total change, despite marking only 20% of the image. They outperform both "
         "comparisons in every layer. The last hidden layer has lower agreement than the others, "
         "but still captures roughly twice the shuffled expectation. The maps therefore contain "
         "information specific to the affected filter, beyond general spatial biases shared by a layer.")
    table(0, ["Hidden layer", "Correct map", "Other filters' maps"], [
        [str(layer + 1), f"{capture[layer]:.1%}", f"{mismatch[layer]:.1%}"] for layer in range(5)])
    text(0, "Read pages 1-4 for the explanation and all six figures. Pages 5-6 retain the precise "
         "definitions, uncertainty calculations, numerical checks, and reproducibility details.")

    heading(1, "2. Does this result hold across images and tiny numerical changes?")
    text(1, "Across images. The plots below show one dot for each of the 32 generated images, "
         "after averaging over filters in that layer. A value of zero means no improvement over "
         "the comparison. For example, capturing 70% instead of 25% is an improvement of 45 "
         "percentage points. Black marks show the average and its 95% confidence interval. "
         "Every layer's average improvement has an interval above zero for both comparisons.")
    figure(1, "spatial_agreement_by_image",
           "Figure 2. Dots show image-level improvements. Left: improvement over shuffled locations. "
           "Right: improvement over other filters' maps. The vertical axis is in percentage points, "
           "not relative percent increases. The black marks summarize images; filters and pixels "
           "are not treated as independent experiments.", 1., 2.65)
    text(1, "Could computer rounding explain the result? Repeating exactly the same model "
         "calculation can produce extremely small numerical differences. We therefore repeated "
         "the comparison after excluding cases with almost no image change. The main conclusion "
         "held at both exclusion thresholds. The earliest layer's agreement became stronger "
         "after these cases were removed.")
    figure(1, "spatial_numerical_sensitivity",
           "Figure 3. Navy includes every positive image change. Purple and grey remove "
           "increasingly tiny changes (mean absolute pixel change below one millionth or one "
           "hundred-thousandth of the raw output range). Improvements remain positive. These "
           "cutoffs check numerical stability; they do not define a perceptually visible effect.", 1., 2.65)

    heading(2, "3. Do later filters contribute finer spatial patterns?")
    text(2, "Spatial frequency describes how quickly a pattern changes from pixel to pixel. "
         "A broad, smooth patch has low frequency; fine texture, stripes, or a checkerboard "
         "has higher frequency. We summarize each signed Shapley Mode with one frequency score, "
         "technically called a normalized spectral centroid. A larger value means finer variation, "
         "not a more important filter or a particular facial feature.")
    figure(2, "layer_spatial_scale",
           "Figure 4. Left: the distribution of frequency scores across every hidden filter and image; "
           "wider sections contain more maps. Right: averages for individual images (grey lines) "
           "and overall averages with uncertainty bars. Purple reduces the influence of the image "
           "edges using a Hann window. This check gives the same broad pattern.", 1., 2.65)
    text(2, f"What we found. Layer averages are {', '.join(f'{s:.3f}' for s in scale)}. "
         "Later layers contain much finer variation overall. However, there is a small dip from "
         f"layer 1 to layer 2, and only {increasing}/32 images increase at every step. It is accurate "
         "to describe an overall shift towards finer patterns, rather than a perfectly ordered "
         "coarse-to-fine progression.")
    figure(2, "output_rgb_spatial_scale",
           f"Figure 5. Final output channels are kept separate. Their mean frequency scores are "
           f"R={rgb['R']:.3f}, G={rgb['G']:.3f}, and B={rgb['B']:.3f}. Dots represent images; "
           "black marks show means and 95% intervals. These channels are not another hidden "
           "layer and are excluded from the trend above.", .55, 2.35)

    heading(3, "4. A clearer example: what should I look for?")
    text(3, "Read each row from left to right. The first two columns show the same image before "
         "and after removing one filter; subtle changes are easier to see in the heatmaps. "
         "Purple shows where the attribution is strongest, and blue shows where pixels actually "
         "changed most. The final column tints the 20% of pixels chosen by the attribution. "
         "Agreement means that the blue changes tend to fall in those selected locations.")
    figure(3, "fixed_spatial_examples",
           "Figure 6. Replacement example: saved image 30 and filters 72, 50, 7, 12, and 8 "
           "in successive hidden layers. The image and candidate filters come from the "
           "pre-existing DCGAN notebook. One filter per layer was selected for visible spatial "
           "structure, without selecting by agreement score. These are illustrations, not a "
           "representative sample. Heatmaps are scaled separately to their own maxima for "
           "display only; purple tint shows the exact selected pixels.", 1., 6.35)
    text(3, "What this does and does not show. These maps help localize a filter's effect, and "
         "later layers tend to contain finer variation. The analysis does not show that filters "
         "represent named facial features. Some fine structure may be texture or checkerboard "
         "patterns. All numerical results still use every saved image and every hidden filter.")

    heading(4, "Technical appendix A: dataset, spatial score, and uncertainty")
    text(4, "Dataset and intervention. One historical, signed, unnormalized set of modes covers "
         "32 fixed latent vectors, 128 x 128 RGB outputs, and 1,987 players: hidden widths "
         "1,024/512/256/128/64 plus three output RGB channels. There are 63,488 hidden "
         "filter-image cases and 96 output-channel cases. The original Shapley game includes "
         "both hidden and output units, so output-channel gating is part of the coalition "
         "context. The notebook specifies 3,000 historical permutations, but this count is "
         "not stored in the averaged pickle. No new Shapley estimates were mixed in.")
    text(4, "For image i, filter u, RGB channel c, and pixel p, let phi be the signed mode, "
         "G the intact generator output, and G with superscript (-u) its single-filter lesion. "
         "The generator is in evaluation mode. Lesions zero the selected output channel after "
         "the full ConvTranspose2d/BatchNorm/ReLU block; the final RGB block uses Sigmoid. "
         "Checkpoint, latent vectors, and zeroing convention match the historical objective.")
    equation(4, r"A_{ui}(p)=\frac{1}{3}\sum_c |\phi_{uic}(p)|,\qquad D_{ui}(p)=\frac{1}{3}\sum_c|G_c(z_i)(p)-G_c^{(-u)}(z_i)(p)|")
    equation(4, r"S_{ui}=\frac{\sum_p M_{ui}(p)D_{ui}(p)}{\sum_p D_{ui}(p)},\qquad |M_{ui}|=\lceil 0.2\times128^2\rceil=3277")
    text(4, "The mask M contains the strongest attribution magnitudes, with seeded random "
         "tie-breaking. Its actual area is 20.00122070%. Controls are 100 uniform spatial "
         "permutations per map, with exact expected capture 3277/16384, and the exact average "
         "capture using every other eligible filter's mask from the same image and layer. "
         "Summed masks implement the latter without sampling pairs. All-zero attribution "
         "maps are ineligible reference masks.")
    text(4, f"Undefined and small effects. Exactly zero total change or all-zero attribution "
         f"makes target agreement undefined; observed counts were {zero_change} and {zero_attr}. "
         f"The primary analysis retains {near_zero:,} positive effects with mean absolute "
         "change below 0.000001. Sensitivity thresholds are 0, 0.000001, and 0.00001 in "
         "raw [0,1] output units. Only target inclusion changes; reference masks are unchanged.")
    text(4, f"Aggregation and intervals. Valid filters are averaged equally within each image/layer; "
         f"images are then averaged equally. The 95% percentile intervals use "
         f"{manifest['settings']['bootstrap']:,} image-bootstrap resamples. Gains are paired "
         "differences within each image, before resampling. The following table reports "
         "capture in percent and gains in percentage points, with 95% intervals.")
    table(4, ["Layer", "Capture (%)", "Gain vs shuffle (pp)", "Gain vs other maps (pp)"], [
        [str(layer + 1), estimate(agreement, layer, "capture", 100, 1),
         estimate(agreement, layer, "gain_vs_shuffle", 100, 1),
         estimate(agreement, layer, "gain_vs_mismatched", 100, 1)] for layer in range(5)])
    text(4, "Interpretation. Shapley Modes average effects over coalitions, whereas the lesion "
         "here starts from the fully intact generator. Perfect equality is not required. "
         "The other-filter comparison addresses spatial biases common to a layer. "
         "No per-filter significance claims or multiple-comparison p-value claims are made.")

    heading(5, "Technical appendix B: frequency definition and reproducibility")
    text(5, "Each signed RGB channel is demeaned separately over spatial positions before "
         "the full two-dimensional FFT. Channel powers are summed, not signed channels, "
         "so opposite RGB signs cannot cancel. DC power is explicitly zeroed. Frequencies "
         "fx and fy are in cycles/pixel, and both sides of the full spectrum are counted.")
    equation(5, r"\widetilde{\phi}_c=\phi_c-\operatorname{mean}_p\phi_c,\quad P(f)=\sum_c|\mathcal{F}(\widetilde{\phi}_c)(f)|^2,\quad P(0)=0")
    equation(5, r"C=\frac{\sum_f \sqrt{f_x^2+f_y^2}\,P(f)}{\sqrt{0.5^2+0.5^2}\sum_fP(f)}")
    text(5, "Normalization uses the two-dimensional Nyquist corner, giving a score from 0 "
         "to 1; high values mean fine spatial variation. Non-DC energy is sum(P)/(H x W), "
         "using Parseval scaling. Exact zero energy receives NaN and is excluded; no "
         "hidden or output cases had zero energy. The primary calculation has no taper. "
         "A separable Hann window is applied after demeaning in the sensitivity analysis, "
         "and its DC power is also removed. Final RGB modes are analyzed separately.")
    table(5, ["Layer", "Standard centroid [95% CI]", "Hann centroid [95% CI]"], [
        [str(layer + 1), estimate(spectral, layer, "spectral_centroid"),
         estimate(spectral, layer, "hann_centroid")] for layer in range(5)])
    text(5, f"Layer trends. The mean within-image last-minus-first difference is "
         f"{float(delta['mean']):.4f} [{float(delta['ci025']):.4f}, {float(delta['ci975']):.4f}]. "
         "The mean image-wise Spearman layer/frequency correlation is 0.9344 "
         "[0.9063, 0.9594]; these are descriptive trends, not evidence for facial semantics. "
         "Image-bootstrap intervals use within-image filter means as above.")
    text(5, f"Compatibility and precision. Summed modes reconstruct intact minus fully lesioned "
         f"outputs with maximum absolute residual {qc['maximum_absolute_residual']:.3g} "
         f"and mean residual {qc['mean_absolute_residual']:.3g}. All 32 image/latent indices "
         f"match, fully lesioned output is zero, and repeated intact inference differs by at "
         f"most {qc['max_repeat_difference']:.3g}. Historical cuDNN/TF32 defaults are retained "
         "(benchmark off; deterministic off; cuDNN TF32 on; matrix-multiply TF32 off). "
         "Inference is float32; modes, metric reductions, and CUDA shuffle reductions use "
         "float64. Seed is 2810; batch size is 32. The CUDA shuffle implementation matches "
         "its CPU reference within 5.56e-16 on the recorded real-size validation.")
    text(5, "Reproduction. From DCGAN, run plot_quantification.py. Then run "
         "build_illustrated_report.py inside quantification to recreate this presentation from "
         "saved results. The original analysis commands are in ../README.md. manifest.json "
         "records artifact SHA-256 values, versions, checkpoint, latents, and perturbations; "
         "unit_mapping.csv and latent_vectors.csv record ordered units and inputs. Figure "
         "layers 1-5 correspond to CSV layers 0-4; filter and image indices remain zero-based. "
         "figure_manifest.json and qualitative_examples.json record styling and example selection. "
         "PDF/SVG figures have editable text; PNGs are exported at 600 dpi. Display normalization "
         "of example heatmaps never enters the scientific measurements.")
    text(5, "Limits. This is one checkpoint and one fixed sample of 32 images. Intervals "
         "do not include training-seed variation or uncertainty in historical Shapley estimates. "
         "Old input hashes were not stored, so compatibility rests on reconstruction and mapping "
         "checks. Frequency results depend on power weighting, resolution, and boundaries; "
         "the Hann analysis probes one boundary choice. No manual semantic annotation is used.")

    expected = {item[1] for page in pages for item in page if item[0] == "figure"}
    available = {p.stem for p in (RESULTS / "figures").glob("*.png")}
    if expected != available:
        raise ValueError(f"Report figure inventory differs: {expected ^ available}")

    def escape(text):
        replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
        return "".join(replacements.get(char, char) for char in text)

    md = [f"# {title}", "", subtitle, ""]
    tex = [r"\documentclass[10pt,a4paper]{article}",
           r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}",
           r"\input{glyphtounicode}", r"\pdfgentounicode=1",
           r"\usepackage[margin=0.7in]{geometry}",
           r"\usepackage{graphicx,booktabs,hyperref,amsmath}", r"\graphicspath{{figures/}}",
           r"\hypersetup{hidelinks,pdftitle={" + escape(title) + r"}}",
           r"\setlength{\parindent}{0pt}", r"\setlength{\parskip}{5pt}",
           r"\setlength{\emergencystretch}{2em}", r"\begin{document}",
           r"{\Large\bfseries " + escape(title) + r"}\par",
           r"{\small " + escape(subtitle) + r"}\par\medskip"]
    for index, page in enumerate(pages):
        if index:
            tex.append(r"\newpage")
        for item in page:
            kind = item[0]
            if kind == "heading":
                md += ["## " + item[1], ""]
                tex.append(r"\subsection*{" + escape(item[1]) + "}")
            elif kind == "text":
                md += [item[1], ""]
                tex.append(escape(item[1]) + "\n")
            elif kind == "equation":
                md += ["$$", item[1], "$$", ""]
                tex.append(r"\[" + item[1] + r"\]")
            elif kind == "figure":
                _, name, caption, width, height = item
                md += [f"![{caption.split('.')[0]}](figures/{name}.png)", "", caption, ""]
                tex += [r"\begin{center}",
                        r"\includegraphics[width=" + str(width) + r"\linewidth,height=" + str(height) + r"in,keepaspectratio]{" + name + ".pdf}",
                        r"\end{center}\vspace{-5pt}",
                        r"{\small " + escape(caption) + r"}\par\medskip"]
            elif kind == "table":
                _, headers, rows = item
                md += ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                md += ["| " + " | ".join(row) + " |" for row in rows]
                md.append("")
                tex += [r"\begin{center}{\small\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + r"}\toprule",
                        " & ".join(escape(h) for h in headers) + r" \\ \midrule"]
                tex += [" & ".join(escape(cell) for cell in row) + r" \\" for row in rows]
                tex.append(r"\bottomrule\end{tabular}}\end{center}")
    tex.append(r"\end{document}")
    md_path = RESULTS / (STEM + ".md")
    tex_path = RESULTS / (STEM + ".tex")
    md_path.write_text("\n".join(md))
    tex_path.write_text("\n".join(tex))
    with tempfile.TemporaryDirectory(prefix="dcgan-illustrated-report-") as tmp:
        command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=" + tmp, tex_path.name]
        for _ in range(2):
            completed = subprocess.run(command, cwd=RESULTS, text=True, capture_output=True)
            if completed.returncode:
                raise RuntimeError(completed.stdout[-8000:] + completed.stderr)
        log = (Path(tmp) / (STEM + ".log")).read_text()
        if "Overfull \\vbox" in log or "Overfull \\hbox" in log:
            raise RuntimeError("Report has overflowing content; adjust figure sizes before publishing.")
        shutil.copyfile(Path(tmp) / (STEM + ".pdf"), RESULTS / (STEM + ".pdf"))
    with zipfile.ZipFile(RESULTS / "PUBLICATION_FIGURES.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted((RESULTS / "figures").iterdir()):
            if path.suffix in (".pdf", ".svg", ".png"):
                bundle.write(path, "figures/" + path.name)
        for name in ("figure_manifest.json", "qualitative_examples.json"):
            bundle.write(RESULTS / name, name)
        bundle.writestr("README.txt", "Six DCGAN figures: vector PDF, editable-text SVG, and 600-dpi PNG.\n"
                        "Colors, sizes, and axes follow the repository's MLP/LLM visual configuration.\n"
                        "Figure layers 1-5 correspond to CSV/model layers 0-4.\n"
                        "The qualitative plate uses notebook image 30; its JSON records selections and display scales.\n"
                        "See ILLUSTRATED_REPORT.pdf alongside this archive for explanations and captions.\n")
    print(RESULTS / (STEM + ".pdf"))
    print(md_path)


if __name__ == "__main__":
    main()
