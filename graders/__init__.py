from .easy_grader import EasyGrader
from .medium_grader import MediumGrader
from .hard_grader import HardGrader
from .analysis_grader import AnalysisGrader

GRADER_MAP = {
    "patch_easy": EasyGrader,
    "patch_medium": MediumGrader,
    "patch_hard": HardGrader,
}
