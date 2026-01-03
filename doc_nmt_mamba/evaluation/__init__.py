"""
Evaluation Framework for Document-Level NMT.

Provides:
- Standard MT metrics (BLEU, chrF, TER, COMET)
- Document-level evaluation (ContraPro pronoun accuracy)
- Named entity recall analysis
- Length sensitivity analysis
- Alignment evaluation (AER with SubwordToWordMapper)

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

from .contrapro import (
    ContrastiveExample,
    ContrastiveResult,
    ContrastivePronounEvaluator,
    ContraProDataset,
    create_synthetic_contrapro_examples,
    evaluate_pronoun_accuracy,
)

from .entity_recall import (
    EntityRecallResult,
    EntityRecallAnalyzer,
    analyze_entity_recall,
)

from .length_analysis import (
    LengthAnalysisResult,
    LengthSensitivityAnalyzer,
    ExtrapolationTester,
    analyze_length_sensitivity,
)

from .alignment import (
    AlignmentResult,
    SubwordToWordMapper,
    AlignmentEvaluator,
    load_awesome_align_alignments,
    evaluate_alignment,
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
    # ContraPro
    "ContrastiveExample",
    "ContrastiveResult",
    "ContrastivePronounEvaluator",
    "ContraProDataset",
    "create_synthetic_contrapro_examples",
    "evaluate_pronoun_accuracy",
    # Entity recall
    "EntityRecallResult",
    "EntityRecallAnalyzer",
    "analyze_entity_recall",
    # Length analysis
    "LengthAnalysisResult",
    "LengthSensitivityAnalyzer",
    "ExtrapolationTester",
    "analyze_length_sensitivity",
    # Alignment
    "AlignmentResult",
    "SubwordToWordMapper",
    "AlignmentEvaluator",
    "load_awesome_align_alignments",
    "evaluate_alignment",
]
