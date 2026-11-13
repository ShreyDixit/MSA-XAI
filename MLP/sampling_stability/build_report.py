#!/usr/bin/env python3
"""Build the illustrated report and figure archive from saved analysis outputs."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
STEM = 'ILLUSTRATED_REPORT'


def read_csv(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def main():
    summary = read_csv('condition_summary.csv')
    contrasts = read_csv('condition_contrasts.csv')
    convergence = read_csv('convergence_summary.csv')
    uncertainty = read_csv('model_uncertainty.csv')
    validation = read_csv('validation.csv')
    analysis = json.loads((OUT/'analysis_manifest.json').read_text())
    manifest = json.loads((OUT/'manifest.json').read_text())
    def row(condition, task):
        return next(r for r in summary if r['condition'] == condition and r['task'] == task)
    def number(condition, task, name):
        return float(row(condition, task)[name])
    def contrast(comparison, task):
        return next(r for r in contrasts if r['comparison'] == comparison and r['task'] == task)
    def f(x):
        return f'{float(x):.4f}'
    tasks = [('overall','Overall'), ('digit_1','Digit 1'), ('digit_9','Digit 9')]
    conditions = ['L1','L2','None']
    widths = [float(r['interval_width']) for r in uncertainty]
    mean_bias = [float(r['bootstrap_bias']) for r in uncertainty]
    max_efficiency = max(float(r['efficiency_error']) for r in validation)
    rechecks = sum(int(r['near_tie_coalition_rechecks']) for r in validation)
    zero_score_denominators = sum(int(r['zero_predicted_count_entries']) for r in validation)
    zero_energies = sum(r['zero_mean_contribution_energy'] == 'True' for r in uncertainty)
    excludes_zero = sum(r['conditional_interval_excludes_zero'] == 'True' for r in contrasts)
    point_outside = sum(float(r['d']) < float(r['ci025']) or float(r['d']) > float(r['ci975']) for r in uncertainty)
    relative_undefined = sum(r['undefined_width_gap_ratio'] == 'True' for r in contrasts)
    title = 'MLP sampling stability of D'
    subtitle = '200-neuron stability check | 30 fixed checkpoints | 1,000 permutations per model'
    pages = [[] for _ in range(5)]
    def heading(page, s): pages[page].append(('heading',s))
    def text(page, s): pages[page].append(('text',s))
    def figure(page, name, caption, height): pages[page].append(('figure',name,caption,height))
    def table(page, headers, rows): pages[page].append(('table',headers,rows))
    def equation(page, s): pages[page].append(('equation',s))

    heading(0, '1. What is being checked?')
    text(0, 'D describes how broadly contributions are spread across hidden neurons. A value near 1 means '
         'that squared contributions are spread fairly evenly; a lower value means that fewer neurons dominate. '
         'Positive and negative contributions both enter through their squares. This experiment asks how much '
         'the estimated D changes when we sample different neuron-removal orders for the same trained network.')
    text(0, 'We use the ten existing 200-neuron checkpoints in each condition: L1, L2 and unregularized (None). '
         'Every checkpoint and every coalition uses the same saved, shuffled batch of 1,024 MNIST test images. '
         'Only the permutation samples vary. The original zero-output intervention and D formula are retained; '
         'no models are retrained. The final 1,000-permutation estimate is a finite-sample reference, not ground truth.')
    figure(0, 'convergence', 'Figure 1. Top: D as the permutation budget increases. Thin lines show individual '
           'models and thick lines show condition means. Bottom: absolute differences from each model\'s '
           '1,000-permutation reference; thick lines show the median difference within a condition. '
           'The final difference is zero by construction. Nested estimates share permutations, so this figure '
           'alone cannot establish convergence to the exact Shapley value.', 3.45)
    ranges = [next(r for r in convergence if r['condition']==c and r['task']=='overall' and int(r['permutations'])==500) for c in conditions]
    text(0, 'At 500 permutations, the median absolute difference from the overall-D reference is '
         + ', '.join(f"{c}: {f(r['median_absolute_difference'])}" for c,r in zip(conditions,ranges)) + '. '
         f'At 1,000, overall mean D is {number("L1","overall","mean_d"):.4f} for L1, '
         f'{number("L2","overall","mean_d"):.4f} for L2 and {number("None","overall","mean_d"):.4f} '
         'without regularization. These overall condition gaps are clearly larger than permutation uncertainty '
         'for these fixed models. Digit-wise estimates are less precise, as the next page shows.')
    text(0, 'Scoring detail. The original evaluation reverses the arguments to the classwise scorer, so its digit '
         'scores are precision values. We preserve this implementation. Overall D is computed from contributions '
         'averaged across those ten scores; it is not D for ordinary pooled test accuracy.')

    heading(1, '2. How uncertain is each model\'s D?')
    text(1, 'For each model, we bootstrap entire permutation vectors and recompute both Shapley contributions '
         'and D. Each interval describes Monte Carlo uncertainty conditional on that checkpoint and this image '
         'batch. The two 500-permutation halves contain independent removal orders; agreement near the dashed '
         'line provides a separate check on reproducibility at that smaller budget.')
    figure(1, 'uncertainty_and_halves', 'Figure 2. Top: one point per checkpoint at 1,000 permutations, with its '
           '95% whole-permutation percentile interval. Conditions are grouped from left to right; checkpoint '
           'indices do not imply pairing between conditions. Bottom: first versus independent second '
           '500-permutation estimates. Each dot is a trained model. The dashed line denotes exact agreement.', 3.8)
    table(1, ['Condition','Score','Mean D','Between-model SD','Median MC width'], [
        [c,label,f(number(c,t,'mean_d')),f(number(c,t,'between_model_sd')),
         f(number(c,t,'median_model_mc_interval_width'))] for t,label in tasks for c in conditions])
    text(1, f'Across all 330 model-by-score estimates, 95% Monte Carlo interval widths range from '
         f'{min(widths):.4f} to {max(widths):.4f}. Between-model SD describes the spread of the ten checkpoint '
         'estimates. It is reported separately and is not an uncertainty bar obtained from permutation replicates. '
         'Digit 1 has wider intervals and larger independent-half differences than digit 9 in each condition.')

    heading(2, '3. Are condition differences larger than sampling uncertainty?')
    text(2, 'We compare condition means while keeping all ten checkpoints per condition fixed. Each bootstrap '
         'draw combines independent permutation resamples within those models. The resulting interval measures '
         'uncertainty from estimating contributions; it does not include training variability or uncertainty '
         'about a population of possible trained models.')
    figure(2, 'condition_comparisons', 'Figure 3. Differences between condition means for overall D and every '
           'digit. Bars are 95% conditional Monte Carlo intervals. A positive value means the first condition '
           'has higher D. Intervals do not resample models or images. These are descriptive comparisons, '
           'with no multiplicity-adjusted population hypothesis tests.', 2.65)
    comparisons = ['L2 minus L1','None minus L1','None minus L2']
    table(2, ['Comparison','Score','Difference','95% conditional MC interval'], [
        [c,label,f(contrast(c,t)['difference_of_model_means']),
         '['+f(contrast(c,t)['conditional_mc_ci025'])+', '+f(contrast(c,t)['conditional_mc_ci975'])+']']
        for t,label in tasks for c in comparisons])
    text(2, f'Across all eleven scores and three comparisons, {excludes_zero}/33 conditional intervals exclude '
         'zero. The exception is L2 versus L1 for digit 6: the difference is 0.0021 with interval '
         '[-0.0087, 0.0123], so this small contrast is unresolved at this budget. '
         'The CSV also reports individual-model interval width relative to each absolute condition gap; '
         'a ratio above 1 means that a typical individual interval is wider than that condition difference. '
         'Larger permutation budgets would not remove genuine variation between trained checkpoints.')
    text(2, f'Nonlinear-estimator caution. Recomputing D after resampling introduces plug-in bias: bootstrap '
         f'mean minus point estimate ranges from {min(mean_bias):.4f} to {max(mean_bias):.4f}; '
         f'{point_outside}/330 point estimates lie outside their percentile intervals. Intervals are approximate '
         'Monte Carlo summaries, not a guarantee that 1,000 permutations recover the exact D. Both the bias '
         'and the independent-half differences are retained in the numerical outputs. The budget remains '
         '1,000 for every model, as requested; no extra sampling is performed.')

    heading(3, '4. Results for every digit')
    figure(3, 'all_digit_convergence', 'Figure 4. All ten digit-score convergence curves. Thin lines are the '
           'ten fixed models in each condition; thick lines are their means. Each panel has its own vertical '
           'scale so that within-digit changes remain visible. Purple: L1; navy: L2; grey: unregularized.', 3.65)
    table(3, ['Score','L1: mean (SD)','L2: mean (SD)','None: mean (SD)'], [
        [label]+[f"{number(c,t,'mean_d'):.4f} ({number(c,t,'between_model_sd'):.4f})" for c in conditions]
        for t,label in [('overall','Overall')]+[(f'digit_{i}',str(i)) for i in range(10)]])
    text(3, 'Parentheses report between-checkpoint sample SD, not Monte Carlo intervals. Every model-by-digit '
         'Monte Carlo interval, half-run comparison and budget estimate is available in the CSVs, including '
         'digits 1 and 9. The analysis is a 200-neuron stability check; it does not establish stability at '
         'other network widths, on other evaluation batches, or for other value functions.')

    heading(4, '5. Technical methods and reproducibility')
    text(4, 'Fixed game. Thirty existing MNISTNet checkpoints (trials 0-9 per condition) are evaluated without '
         'retraining in float32 on cuda:2, with TF32 matrix multiplication disabled. Images are flattened and '
         'divided by 255 as in the original code. The batch is sampled once without replacement from the '
         f'10,000 test images; batch seed {manifest["identity"]["batch_seed"]}, master seed 2810. '
         'True digit counts for 0-9 are '+str(manifest['digit_counts'])+'. '
         'Each hidden-neuron output is set to zero immediately after layer1, before LeakyReLU. '
         'The output bias remains active when all hidden neurons are removed.')
    text(4, 'Preserved score. Let n_c be the number of images predicted as digit c and TP_c the correctly '
         'predicted images of that digit. The original argument order in evaluate_model makes the '
         'confusion-matrix row totals equal n_c, so the implemented score is:')
    equation(4, r'v_c(S)=\frac{\mathrm{TP}_c(S)}{n_c(S)+10^{-6}}.')
    text(4, 'This is precision, with zero for an unpredicted digit. It is not recall or pooled accuracy. '
         'All ten digit scores are measured jointly for each coalition. For permutation r, remove neurons '
         'in the sampled order and record the drop in every digit score at each removal, assigning that '
         'vector to the removed neuron. This yields X with shape (1000, 200, 10) per model, stored in full. '
         'The orders use independent seeded NumPy PCG64 streams across models; there is no antithetic pairing. '
         'Prefixes 1:R define budgets 100, 250, 500, 750 and 1000; the halves are rows 1:500 and 501:1000.')
    equation(4, r'\widehat\phi_{i,c}^{(R)}=R^{-1}\sum_{r=1}^{R}X_{r,i,c},\qquad'
         r'\widehat\phi_{i,\mathrm{overall}}=10^{-1}\sum_{c=0}^{9}\widehat\phi_{i,c}.')
    equation(4, r'p_i=\frac{\widehat\phi_i^2}{\sum_j\widehat\phi_j^2},\qquad '
         r'D=-\frac{\sum_{i=1}^{200}p_i\log p_i}{\log 200}.')
    text(4, 'As in the original function, D=0 if all contributions are zero; exactly zero p entries become '
         '1e-10 before taking logs, and D is clipped to [0,1]. Averaging contributions precedes squaring and '
         'entropy; we do not average digit-wise D to obtain overall D. Ratios with zero denominators are '
         f'recorded as undefined, not replaced by finite values. Here there are {zero_energies} zero-energy '
         f'mean vectors, {zero_score_denominators:,} coalition-score entries with n_c=0 (handled by the '
         f'original epsilon rule), and {relative_undefined} undefined condition-gap ratios.')
    text(4, f'Bootstrap. For each model, {analysis["bootstrap_resamples"]:,} independent multinomial weight '
         'vectors resample 1,000 whole permutation rows with replacement. The same weights apply jointly '
         'to all 200 neurons and ten digit tasks. Recompute mean signed contributions, their overall average '
         'and then D; report the 2.5th and 97.5th percentiles and bootstrap SD. Neurons, tasks and models '
         'are not independent permutation replicates. Model-mean intervals average the ten independent '
         'within-model bootstrap draws without resampling models. Between-model SD uses ddof=1.')
    text(4, f'Numerical checks. Batched coalition evaluation precomputes intact hidden activations and applies '
         'the same zero masks and final affine layer. LeakyReLU(0)=0 makes this equivalent to the original '
         'intervention. Coalitions with a top-two logit margin below 1e-4 are re-evaluated through the '
         f'original forward path ({rechecks:,} rechecks). All 128 validation coalitions per model exactly '
         'match the original sklearn-based score (3,840 coalitions, zero mismatches). The maximum '
         f'permutation efficiency residual is {max_efficiency:.2e}. Five known-answer statistical tests '
         'verify D, task averaging, score denominators, marginal assignment and joint resampling.')
    text(4, 'Artifacts. evaluation_batch.npz saves indices, images and labels; permutations_*.npz saves '
         'orders, marginal arrays, coalition correct/predicted counts and provenance. seeds.csv and '
         'manifest.json record model hashes and all seeds; analysis_manifest.json and figure_manifest.json '
         'record analysis settings, source and output hashes. CSVs separate model uncertainty, condition '
         'contrasts, checkpoint variation, contribution means/SEs and validation. Reproduce with '
         'run_sampling_stability.py, analyze_sampling_stability.py, plot_sampling_stability.py and '
         'sampling_stability/build_report.py, in that order; see README.md for commands.')

    def escape(s):
        replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#',
            '_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
            '−':'-'}
        return ''.join(replacements.get(c,c) for c in s)
    md=[f'# {title}', '', subtitle, '']
    tex=[r'\documentclass[10pt,a4paper]{article}',r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern}',
         r'\input{glyphtounicode}',r'\pdfgentounicode=1',r'\usepackage[margin=0.65in]{geometry}',
         r'\usepackage{graphicx,booktabs,hyperref,amsmath}',r'\graphicspath{{figures/}}',
         r'\hypersetup{hidelinks,pdftitle={'+escape(title)+r'}}',r'\setlength{\parindent}{0pt}',
         r'\setlength{\parskip}{5pt}',r'\setlength{\emergencystretch}{3em}',r'\begin{document}',
         r'{\Large\bfseries '+escape(title)+r'}\par',r'{\small '+escape(subtitle)+r'}\par']
    for page_index,page in enumerate(pages):
        if page_index: tex.append(r'\newpage')
        for item in page:
            kind=item[0]
            if kind=='heading':
                md += ['## '+item[1],'']
                tex.append(r'\subsection*{'+escape(item[1])+'}')
            elif kind=='text':
                md += [item[1],'']
                tex.append(escape(item[1])+'\n')
            elif kind=='equation':
                md += ['$$',item[1],'$$','']
                tex.append(r'\['+item[1]+r'\]')
            elif kind=='figure':
                _,name,caption,height=item
                md += [f'![{caption.split(".")[0]}](figures/{name}.png)','',caption,'']
                tex += [r'\begin{center}\includegraphics[width=\linewidth,height='+str(height)+
                    r'in,keepaspectratio]{'+name+r'.pdf}\end{center}\vspace{-4pt}',
                    r'{\small '+escape(caption)+r'}\par']
            elif kind=='table':
                _,headers,rows=item
                md += ['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
                md += ['| '+' | '.join(r)+' |' for r in rows]+['']
                tex += [r'\begin{center}{\small\begin{tabular}{l'+'r'*(len(headers)-1)+r'}\toprule',
                    ' & '.join(escape(h) for h in headers)+r' \\ \midrule']
                tex += [' & '.join(escape(v) for v in r)+r' \\' for r in rows]
                tex.append(r'\bottomrule\end{tabular}}\end{center}')
    tex.append(r'\end{document}')
    (OUT/(STEM+'.md')).write_text('\n'.join(md))
    (OUT/(STEM+'.tex')).write_text('\n'.join(tex))
    with tempfile.TemporaryDirectory(prefix='mlp-stability-report-') as directory:
        for _ in range(2):
            completed=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',
                '-output-directory='+directory,STEM+'.tex'], cwd=OUT, capture_output=True,text=True)
            if completed.returncode: raise RuntimeError(completed.stdout[-6000:])
        log=(Path(directory)/(STEM+'.log')).read_text()
        if 'Overfull \\vbox' in log or 'Overfull \\hbox' in log:
            raise RuntimeError('Report content overflows; revise layout.\n'+log[-4000:])
        shutil.copyfile(Path(directory)/(STEM+'.pdf'),OUT/(STEM+'.pdf'))
    expected={item[1] for page in pages for item in page if item[0]=='figure'}
    assert expected=={p.stem for p in (OUT/'figures').glob('*.png')}
    with zipfile.ZipFile(OUT/'PUBLICATION_FIGURES.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted((OUT/'figures').iterdir()):
            if path.suffix in {'.pdf','.svg','.png'}: z.write(path,'figures/'+path.name)
        z.write(OUT/'figure_manifest.json','figure_manifest.json')
        z.writestr('README.txt','Four figures: vector PDF, editable-text SVG, 600-dpi PNG. '
            'Captions and explanations are in ILLUSTRATED_REPORT.pdf. Colors follow MLP/visual_config.py.\n')
    with zipfile.ZipFile(OUT/'NUMERICAL_SUMMARIES.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.glob('*.csv')):
            z.write(path,path.name)
        for path in sorted(OUT.glob('*manifest.json')):
            z.write(path,path.name)
        z.write(OUT/'evaluation_batch.npz','evaluation_batch.npz')
        z.write(ROOT/'README.md','README.md')
        z.write(ROOT/'requirements.txt','requirements.txt')
        z.writestr('RAW_ARRAYS.txt','The full 30 permutation-vector archives and 30 bootstrap '
            'archives are retained separately in MLP/sampling_stability/results/. '
            'See README.md for shapes and ordering.\n')
    print(OUT/(STEM+'.pdf'))


if __name__=='__main__': main()
