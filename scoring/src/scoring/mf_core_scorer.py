from typing import Dict, List, Optional

from . import constants as c
from .mf_base_scorer import MFBaseScorer


class MFCoreScorer(MFBaseScorer):
  def __init__(
    self,
    seed: Optional[int] = None,
    pseudoraters: Optional[bool] = False,
    useStableInitialization: bool = True,
    saveIntermediateState: bool = False,
    threads: int = c.defaultNumThreads,
    firmRejectThreshold: Optional[float] = 0.3,
    minMinorityNetHelpfulRatings: Optional[int] = 4,
    minMinorityNetHelpfulRatio: Optional[float] = 0.05,
  ) -> None:
    """Configure MFCoreScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
      pseudoraters: if True, compute optional pseudorater confidence intervals
      threads: number of threads to use for intra-op parallelism in pytorch
    """
    super().__init__(
      excludeTopics=True,
      includedGroups=c.coreGroups,
      includeUnassigned=True,
      captureThreshold=0.5,
      seed=seed,
      pseudoraters=pseudoraters,
      useStableInitialization=useStableInitialization,
      saveIntermediateState=saveIntermediateState,
      threads=threads,
      firmRejectThreshold=firmRejectThreshold,
      minMinorityNetHelpfulRatings=minMinorityNetHelpfulRatings,
      minMinorityNetHelpfulRatio=minMinorityNetHelpfulRatio,
    )

  def get_name(self):
    return "MFCoreScorer"

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    mapping = {
      c.internalNoteInterceptKey: c.coreNoteInterceptKey,
      c.internalNoteFactor1Key: c.coreNoteFactor1Key,
      c.internalRatingStatusKey: c.coreRatingStatusKey,
      c.internalActiveRulesKey: c.coreActiveRulesKey,
      c.noteInterceptMinKey: c.coreNoteInterceptMinKey,
      c.noteInterceptMaxKey: c.coreNoteInterceptMaxKey,
      c.numFinalRoundRatingsKey: c.coreNumFinalRoundRatingsKey,
      c.lowDiligenceNoteInterceptKey: c.lowDiligenceLegacyNoteInterceptKey,
      c.internalNoteInterceptNoHighVolKey: c.coreNoteInterceptNoHighVolKey,
      c.internalNoteInterceptNoCorrelatedKey: c.coreNoteInterceptNoCorrelatedKey,
      c.internalNoteInterceptPopulationSampledKey: c.coreNoteInterceptPopulationSampledKey,
      c.negFactorPopulationSampledRatingCountKey: c.coreNegFactorPopulationSampledRatingCountKey,
      c.posFactorPopulationSampledRatingCountKey: c.corePosFactorPopulationSampledRatingCountKey,
    }
    # Add dynamic mappings for Factor2+ (keep internal names for analysis)
    for i in range(2, self._get_num_factors() + 1):
      mapping[c.note_factor_key(i)] = c.note_factor_key(i)
    return mapping

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    mapping = {
      c.internalRaterInterceptKey: c.coreRaterInterceptKey,
      c.internalRaterFactor1Key: c.coreRaterFactor1Key,
      c.internalFirstRoundRaterInterceptKey: c.coreFirstRoundRaterInterceptKey,
      c.internalFirstRoundRaterFactor1Key: c.coreFirstRoundRaterFactor1Key,
    }
    # Add dynamic mappings for Factor2+ (keep internal names for analysis)
    for i in range(2, self._get_num_factors() + 1):
      mapping[c.rater_factor_key(i)] = c.rater_factor_key(i)
    return mapping

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    cols = [
      c.noteIdKey,
      c.coreNoteInterceptKey,
      c.coreNoteFactor1Key,
    ]
    # Add Factor2+ columns dynamically for SIGreg analysis
    for i in range(2, self._get_num_factors() + 1):
      cols.append(c.note_factor_key(i))
    cols.extend([
      c.coreRatingStatusKey,
      c.coreActiveRulesKey,
      c.activeFilterTagsKey,
      c.coreNoteInterceptMinKey,
      c.coreNoteInterceptMaxKey,
      c.coreNumFinalRoundRatingsKey,
      c.coreNoteInterceptNoHighVolKey,
      c.coreNoteInterceptNoCorrelatedKey,
      c.coreNoteInterceptPopulationSampledKey,
    ])
    return cols

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    cols = [
      c.raterParticipantIdKey,
      c.coreRaterInterceptKey,
      c.coreRaterFactor1Key,
    ]
    # Add Factor2+ columns dynamically for SIGreg analysis
    for i in range(2, self._get_num_factors() + 1):
      cols.append(c.rater_factor_key(i))
    cols.extend([
      c.crhCrnhRatioDifferenceKey,
      c.meanNoteScoreKey,
      c.raterAgreeRatioKey,
      c.aboveHelpfulnessThresholdKey,
      c.coreFirstRoundRaterInterceptKey,
      c.coreFirstRoundRaterFactor1Key,
    ])
    return cols

  def get_auxiliary_note_info_cols(self) -> List[str]:
    base = super().get_auxiliary_note_info_cols()
    # Include core-prefixed per-sign population-sampled counts in core model auxiliary output only
    return base + [
      c.coreNegFactorPopulationSampledRatingCountKey,
      c.corePosFactorPopulationSampledRatingCountKey,
    ]
