"""
Publication-Quality Visualization for Document-Level NMT.

Generates figures for NeurIPS/ICML/ACL papers:
- Figure 2: Throughput vs. Sequence Length (Log-Log overlay)
- Figure 3: Memory vs. Sequence Length
- Figure 4: ContraPro Accuracy vs. Distance
- Figure 5: Latency breakdown (TTFT/ITL)
- Table 1: Quality metrics (LaTeX)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# Publication-quality plot settings
PLOT_STYLE = {
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
}

# Color scheme for models
COLORS = {
    'baseline': '#E24A33',  # Red/Orange for Transformer
    'thesis': '#348ABD',    # Blue for Mamba
    'random': '#988ED5',    # Purple for random baseline
}

MARKERS = {
    'baseline': 's',  # Square
    'thesis': 'o',    # Circle
}


@dataclass
class ModelResults:
    """Aggregated results from a model evaluation."""
    name: str
    label: str

    # Quality metrics
    bleu: Optional[float] = None
    bleu_ci: Optional[Tuple[float, float]] = None
    comet: Optional[float] = None
    comet_ci: Optional[Tuple[float, float]] = None
    chrf: Optional[float] = None

    # Efficiency data (lists indexed by seq_len)
    seq_lengths: Optional[List[int]] = None
    throughput: Optional[List[float]] = None
    memory_gb: Optional[List[float]] = None
    ttft_ms: Optional[List[float]] = None
    itl_ms: Optional[List[float]] = None

    # ContraPro data
    contrapro_accuracy: Optional[float] = None
    contrapro_by_distance: Optional[Dict[str, float]] = None


def load_results(json_path: str, label: str) -> ModelResults:
    """Load evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = ModelResults(
        name=Path(json_path).stem,
        label=label,
    )

    # Quality metrics
    if 'quality' in data.get('results', data):
        q = data.get('results', data).get('quality', {})
        results.bleu = q.get('bleu')
        results.comet = q.get('comet')
        results.chrf = q.get('chrf')
        if q.get('bleu_ci_low') and q.get('bleu_ci_high'):
            results.bleu_ci = (q['bleu_ci_low'], q['bleu_ci_high'])
        if q.get('comet_ci_low') and q.get('comet_ci_high'):
            results.comet_ci = (q['comet_ci_low'], q['comet_ci_high'])

    # Efficiency metrics
    if 'efficiency' in data.get('results', data):
        e = data.get('results', data).get('efficiency', {})
        if 'seq_lengths' in e:
            results.seq_lengths = e['seq_lengths']
            results.throughput = e.get('throughput', e.get('tokens_per_second'))
            results.memory_gb = e.get('memory_gb', e.get('peak_memory_gb'))
            results.ttft_ms = e.get('ttft_ms', e.get('time_to_first_token_ms'))
            results.itl_ms = e.get('itl_ms', e.get('inter_token_latency_ms'))

    # ContraPro metrics
    if 'contrapro' in data.get('results', data):
        c = data.get('results', data).get('contrapro', {})
        results.contrapro_accuracy = c.get('accuracy')
        results.contrapro_by_distance = c.get('accuracy_by_distance')

    return results


def load_efficiency_csv(csv_path: str, label: str) -> ModelResults:
    """Load efficiency results from CSV file."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required for CSV loading")

    df = pd.read_csv(csv_path)
    df_bs1 = df[df['batch_size'] == 1].sort_values('seq_len')

    return ModelResults(
        name=Path(csv_path).stem,
        label=label,
        seq_lengths=df_bs1['seq_len'].tolist(),
        throughput=df_bs1['tokens_per_second'].tolist(),
        memory_gb=df_bs1['peak_memory_gb'].tolist(),
        ttft_ms=df_bs1['time_to_first_token_ms'].tolist(),
        itl_ms=df_bs1['inter_token_latency_ms'].tolist(),
    )


def plot_throughput_comparison(
    baseline: ModelResults,
    thesis: ModelResults,
    output_path: Path,
) -> None:
    """Generate Figure 2: Throughput vs. Sequence Length (Log-Log)."""
    if not PLOTTING_AVAILABLE:
        return

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    if baseline.seq_lengths and baseline.throughput:
        ax.loglog(
            baseline.seq_lengths, baseline.throughput,
            marker=MARKERS['baseline'], color=COLORS['baseline'],
            linewidth=2.5, markersize=10,
            label=baseline.label,
        )

    if thesis.seq_lengths and thesis.throughput:
        ax.loglog(
            thesis.seq_lengths, thesis.throughput,
            marker=MARKERS['thesis'], color=COLORS['thesis'],
            linewidth=2.5, markersize=10,
            label=thesis.label,
        )

    # Theoretical scaling lines
    if thesis.seq_lengths:
        x = np.array(thesis.seq_lengths)
        y_quad = thesis.throughput[0] * (thesis.seq_lengths[0] / x) ** 2
        ax.loglog(x, y_quad, '--', color='gray', alpha=0.5, label=r'$O(L^2)$ scaling')
        y_lin = thesis.throughput[0] * (thesis.seq_lengths[0] / x)
        ax.loglog(x, y_lin, ':', color='gray', alpha=0.5, label=r'$O(L)$ scaling')

    ax.set_xlabel('Sequence Length (tokens)', fontsize=14)
    ax.set_ylabel('Throughput (tokens/second)', fontsize=14)
    ax.set_title('Inference Throughput vs. Sequence Length\n(Batch Size = 1, H100 80GB)', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()
    plt.savefig(output_path / 'figure2_throughput.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure2_throughput.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_memory_comparison(
    baseline: ModelResults,
    thesis: ModelResults,
    output_path: Path,
) -> None:
    """Generate memory consumption comparison plot."""
    if not PLOTTING_AVAILABLE:
        return

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    if baseline.seq_lengths and baseline.memory_gb:
        ax.plot(
            baseline.seq_lengths, baseline.memory_gb,
            marker=MARKERS['baseline'], color=COLORS['baseline'],
            linewidth=2.5, markersize=10,
            label=baseline.label,
        )

    if thesis.seq_lengths and thesis.memory_gb:
        ax.plot(
            thesis.seq_lengths, thesis.memory_gb,
            marker=MARKERS['thesis'], color=COLORS['thesis'],
            linewidth=2.5, markersize=10,
            label=thesis.label,
        )

    ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
               label='H100 80GB Limit')

    ax.set_xlabel('Sequence Length (tokens)', fontsize=14)
    ax.set_ylabel('Peak GPU Memory (GB)', fontsize=14)
    ax.set_title('Memory Consumption vs. Sequence Length\n(Batch Size = 1)', fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 85)

    plt.tight_layout()
    plt.savefig(output_path / 'figure3_memory.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure3_memory.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_contrapro_comparison(
    baseline: ModelResults,
    thesis: ModelResults,
    output_path: Path,
) -> None:
    """Generate Figure 4: ContraPro Accuracy vs. Distance."""
    if not PLOTTING_AVAILABLE:
        return

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    bucket_centers = {
        "0-50": 25, "51-100": 75, "101-200": 150,
        "201-500": 350, "501-1000": 750, "1000+": 1500
    }

    def extract_xy(by_distance: Dict) -> Tuple[List, List]:
        x, y = [], []
        for bucket, acc in sorted(by_distance.items(), key=lambda kv: bucket_centers.get(kv[0], 0)):
            if acc is not None and bucket in bucket_centers:
                x.append(bucket_centers[bucket])
                y.append(acc * 100)
        return x, y

    if baseline.contrapro_by_distance:
        x, y = extract_xy(baseline.contrapro_by_distance)
        ax.plot(x, y, marker=MARKERS['baseline'], color=COLORS['baseline'],
               linewidth=2.5, markersize=10, label=baseline.label)

    if thesis.contrapro_by_distance:
        x, y = extract_xy(thesis.contrapro_by_distance)
        ax.plot(x, y, marker=MARKERS['thesis'], color=COLORS['thesis'],
               linewidth=2.5, markersize=10, label=thesis.label)

    ax.axhline(y=50, color=COLORS['random'], linestyle='--', alpha=0.7,
               linewidth=1.5, label='Random Baseline (50%)')

    ax.set_xlabel('Antecedent Distance (tokens)', fontsize=14)
    ax.set_ylabel('Pronoun Disambiguation Accuracy (%)', fontsize=14)
    ax.set_title('Context Utilization: Accuracy vs. Antecedent Distance', fontsize=14)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 100)
    ax.set_xlim(0, 1600)

    plt.tight_layout()
    plt.savefig(output_path / 'figure4_contrapro.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure4_contrapro.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_latency_breakdown(
    baseline: ModelResults,
    thesis: ModelResults,
    output_path: Path,
) -> None:
    """Generate TTFT vs ITL breakdown comparison."""
    if not PLOTTING_AVAILABLE:
        return

    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TTFT (Pre-fill)
    if baseline.seq_lengths and baseline.ttft_ms:
        axes[0].plot(baseline.seq_lengths, baseline.ttft_ms,
                    marker=MARKERS['baseline'], color=COLORS['baseline'],
                    linewidth=2.5, markersize=8, label=baseline.label)
    if thesis.seq_lengths and thesis.ttft_ms:
        axes[0].plot(thesis.seq_lengths, thesis.ttft_ms,
                    marker=MARKERS['thesis'], color=COLORS['thesis'],
                    linewidth=2.5, markersize=8, label=thesis.label)

    axes[0].set_xlabel('Sequence Length', fontsize=12)
    axes[0].set_ylabel('Time to First Token (ms)', fontsize=12)
    axes[0].set_title('Pre-fill Latency (TTFT)', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # ITL (Decoding)
    if baseline.seq_lengths and baseline.itl_ms:
        axes[1].plot(baseline.seq_lengths, baseline.itl_ms,
                    marker=MARKERS['baseline'], color=COLORS['baseline'],
                    linewidth=2.5, markersize=8, label=baseline.label)
    if thesis.seq_lengths and thesis.itl_ms:
        axes[1].plot(thesis.seq_lengths, thesis.itl_ms,
                    marker=MARKERS['thesis'], color=COLORS['thesis'],
                    linewidth=2.5, markersize=8, label=thesis.label)

    axes[1].set_xlabel('Sequence Length', fontsize=12)
    axes[1].set_ylabel('Inter-Token Latency (ms)', fontsize=12)
    axes[1].set_title('Decoding Latency (ITL)', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'figure5_latency.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure5_latency.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_quality_table(
    baseline: ModelResults,
    thesis: ModelResults,
    output_path: Path,
) -> str:
    """Generate Table 1: Translation Quality Comparison (LaTeX)."""

    def fmt_metric(val, ci=None):
        if val is None:
            return "N/A"
        if ci:
            return f"{val:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
        return f"{val:.2f}"

    def fmt_comet(val, ci=None):
        if val is None:
            return "N/A"
        return f"{val:.4f}"

    bleu_winner = None
    if baseline.bleu and thesis.bleu:
        bleu_winner = 'thesis' if thesis.bleu > baseline.bleu else 'baseline'

    comet_winner = None
    if baseline.comet and thesis.comet:
        comet_winner = 'thesis' if thesis.comet > baseline.comet else 'baseline'

    latex = r"""
\begin{table}[t]
\centering
\caption{Translation Quality on IWSLT14 De$\to$En Test Set.
Best scores in \textbf{bold}. 95\% confidence intervals from paired bootstrap resampling.}
\label{tab:quality}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Params} & \textbf{BLEU} ($\uparrow$) & \textbf{COMET} ($\uparrow$) & \textbf{ChrF++} ($\uparrow$) \\
\midrule
"""

    bleu_base = fmt_metric(baseline.bleu, baseline.bleu_ci)
    comet_base = fmt_comet(baseline.comet, baseline.comet_ci)
    chrf_base = fmt_metric(baseline.chrf)

    if bleu_winner == 'baseline':
        bleu_base = f"\\textbf{{{bleu_base}}}"
    if comet_winner == 'baseline':
        comet_base = f"\\textbf{{{comet_base}}}"

    latex += f"{baseline.label} & 77M & {bleu_base} & {comet_base} & {chrf_base} \\\\\n"

    bleu_thesis = fmt_metric(thesis.bleu, thesis.bleu_ci)
    comet_thesis = fmt_comet(thesis.comet, thesis.comet_ci)
    chrf_thesis = fmt_metric(thesis.chrf)

    if bleu_winner == 'thesis':
        bleu_thesis = f"\\textbf{{{bleu_thesis}}}"
    if comet_winner == 'thesis':
        comet_thesis = f"\\textbf{{{comet_thesis}}}"

    latex += f"{thesis.label} & 77M & {bleu_thesis} & {comet_thesis} & {chrf_thesis} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    table_path = output_path / 'table1_quality.tex'
    with open(table_path, 'w') as f:
        f.write(latex)

    return latex


def generate_all_figures(
    baseline_path: str,
    thesis_path: str,
    output_dir: str,
    baseline_label: str = "Transformer Baseline",
    thesis_label: str = "Hybrid Mamba-Attention",
) -> None:
    """Generate all publication figures from evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    baseline = load_results(baseline_path, baseline_label)
    thesis = load_results(thesis_path, thesis_label)

    print("Generating figures...")

    if baseline.seq_lengths or thesis.seq_lengths:
        plot_throughput_comparison(baseline, thesis, output_path)
        print("  Saved: figure2_throughput.pdf")
        plot_memory_comparison(baseline, thesis, output_path)
        print("  Saved: figure3_memory.pdf")
        plot_latency_breakdown(baseline, thesis, output_path)
        print("  Saved: figure5_latency.pdf")

    if baseline.contrapro_by_distance or thesis.contrapro_by_distance:
        plot_contrapro_comparison(baseline, thesis, output_path)
        print("  Saved: figure4_contrapro.pdf")

    generate_quality_table(baseline, thesis, output_path)
    print("  Saved: table1_quality.tex")

    print(f"\nAll artifacts saved to: {output_path}")
