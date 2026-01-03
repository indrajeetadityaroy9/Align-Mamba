"""
Evaluation Framework for Document-Level NMT.

Consolidated structure:
- metrics.py: BLEU, chrF, TER, COMET scorers
- analysis.py: ContraPro, Entity, Alignment, Length analysis
- runner.py: Unified evaluation pipeline orchestration
- visualization.py: Publication-quality plotting

Key evaluation protocols:
1. Standard MT: BLEU + COMET on test set
2. Pronoun Resolution: ContraPro-style contrastive scoring (target: +10pp over baseline)
3. Entity Coherence: Recall of NEs from source in translation
4. Length Extrapolation: Test at 2x-4x training sequence length
5. Alignment Quality: AER < 0.30 (competitive with neural MT)
"""

from .metrics import (
    EvaluationResult,
    BLEUScorer,
    CHRFScorer,
    TERScorer,
    COMETScorer,
    EvaluationSuite,
    compute_bleu,
    compute_comet,
)

from .analysis import (
    # Alignment
    AlignmentResult,
    SubwordToWordMapper,
    AlignmentEvaluator,
    load_awesome_align_alignments,
    # ContraPro
    ContrastiveExample,
    ContrastiveResult,
    ContrastivePronounEvaluator,
    ContraProDataset,
    evaluate_pronoun_accuracy,
    # Entity recall
    EntityRecallResult,
    EntityRecallAnalyzer,
    SimpleNER,
    SpaCyNER,
    analyze_entity_recall,
    # Length analysis
    LengthAnalysisResult,
    LengthSensitivityAnalyzer,
    ExtrapolationTester,
    analyze_length_sensitivity,
)

from .runner import (
    RunnerConfig,
    QualityResult,
    EfficiencyResult,
    ContraProResult,
    FullEvaluationResult,
    EvaluationRunner,
    check_dependencies,
)

from .visualization import (
    ModelResults,
    load_results,
    load_efficiency_csv,
    plot_throughput_comparison,
    plot_memory_comparison,
    plot_contrapro_comparison,
    plot_latency_breakdown,
    generate_quality_table,
    generate_all_figures,
)

__all__ = [
    # Metrics
    "EvaluationResult",
    "BLEUScorer",
    "CHRFScorer",
    "TERScorer",
    "COMETScorer",
    "EvaluationSuite",
    "compute_bleu",
    "compute_comet",
    # Alignment
    "AlignmentResult",
    "SubwordToWordMapper",
    "AlignmentEvaluator",
    "load_awesome_align_alignments",
    # ContraPro
    "ContrastiveExample",
    "ContrastiveResult",
    "ContrastivePronounEvaluator",
    "ContraProDataset",
    "evaluate_pronoun_accuracy",
    # Entity recall
    "EntityRecallResult",
    "EntityRecallAnalyzer",
    "SimpleNER",
    "SpaCyNER",
    "analyze_entity_recall",
    # Length analysis
    "LengthAnalysisResult",
    "LengthSensitivityAnalyzer",
    "ExtrapolationTester",
    "analyze_length_sensitivity",
    # Runner
    "RunnerConfig",
    "QualityResult",
    "EfficiencyResult",
    "ContraProResult",
    "FullEvaluationResult",
    "EvaluationRunner",
    "check_dependencies",
    # Visualization
    "ModelResults",
    "load_results",
    "load_efficiency_csv",
    "plot_throughput_comparison",
    "plot_memory_comparison",
    "plot_contrapro_comparison",
    "plot_latency_breakdown",
    "generate_quality_table",
    "generate_all_figures",
]
