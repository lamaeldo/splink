from __future__ import annotations

from typing import Any

from splink.internals.em_training_session import EMTrainingSession
from splink.internals.input_column import InputColumn
from splink.internals.pairwise_prior import PairwisePriorParameters
from splink.internals.pairwise_prior_em import (
    FREQUENCY_COLUMN,
    expectation_maximisation_with_pairwise_prior,
)
from splink.internals.pipeline import CTEPipeline
from splink.internals.settings import CoreModelSettings
from splink.internals.splink_dataframe import SplinkDataFrame
from splink.internals.vertically_concatenate import enqueue_df_concat


class PairwisePriorTrainingSession(EMTrainingSession):
    """EM diagnostics plus the history of a frequency-dependent training prior.

    The prior applies only within this blocked training session. Prediction
    continues to use the linker's global prior. Existing EM history charts show
    the mean prior; ``prior_history`` describes its variation across pairs.
    """

    _training_name = "Pairwise prior EM"
    _can_estimate_global_prior = False

    def __init__(
        self,
        *args: Any,
        blocking_columns: list[InputColumn],
        prior_coefficient_bound: float,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if not self.core_model_settings.comparisons:
            raise ValueError(
                "The blocking columns consume every comparison. Pairwise-prior "
                "EM needs at least one comparison remaining after blocking."
            )
        self._blocking_columns = blocking_columns
        self._prior_coefficient_bound = prior_coefficient_bound
        self._prior_history: list[PairwisePriorParameters] = []
        self._owned_vectors: SplinkDataFrame | None = None
        self.converged = False

    @property
    def prior_history(self) -> list[PairwisePriorParameters]:
        """A copy of the prior's iteration history, including initial values."""
        return list(self._prior_history)

    def _comparison_vectors(self) -> SplinkDataFrame:
        # Reuse the native blocking/sampling and comparison pipeline. Frequencies
        # come from the complete input, independently of the sampled candidate set.
        vectors = super()._comparison_vectors()
        try:
            enriched = self._add_block_frequencies(vectors)
        finally:
            vectors.drop_table_from_database_and_remove_from_cache()
        enriched.templated_name = "__splink__df_comparison_vectors"
        self._owned_vectors = enriched
        return enriched

    def _add_block_frequencies(self, vectors: SplinkDataFrame) -> SplinkDataFrame:
        pipeline = CTEPipeline([vectors])
        enqueue_df_concat(self._original_linker, pipeline)
        column_info = self._original_linker._settings_obj.column_info_settings
        source = column_info.source_dataset_input_column
        partition_columns = [c.name for c in self._blocking_columns]
        if source is not None:
            partition_columns.insert(0, source.name)
        identifiers = ", ".join(col.name for col in self.unique_id_input_columns)
        pipeline.enqueue_sql(
            f"select {identifiers}, count(*) over (partition by "
            + ", ".join(partition_columns)
            + f") as {FREQUENCY_COLUMN} from __splink__df_concat",
            "__splink__records_with_block_frequency",
        )
        left_join = " and ".join(
            f"v.{col.name_l} = l.{col.name}" for col in self.unique_id_input_columns
        )
        right_join = " and ".join(
            f"v.{col.name_r} = r.{col.name}" for col in self.unique_id_input_columns
        )
        # CASE rather than GREATEST keeps this portable to SQLite as well.
        frequency = (
            f"case when l.{FREQUENCY_COLUMN} >= r.{FREQUENCY_COLUMN} "
            f"then l.{FREQUENCY_COLUMN} else r.{FREQUENCY_COLUMN} end"
        )
        pipeline.enqueue_sql(
            f"select v.*, {frequency} as {FREQUENCY_COLUMN} "
            f"from {vectors.physical_name} as v "
            "inner join __splink__records_with_block_frequency as l "
            f"on {left_join} "
            "inner join __splink__records_with_block_frequency as r "
            f"on {right_join}",
            "__splink__df_comparison_vectors_with_prior",
        )
        return self.db_api.sql_pipeline_to_splink_dataframe(pipeline)

    def _train(self, cvv: SplinkDataFrame | None = None) -> CoreModelSettings:
        try:
            return super()._train(cvv)
        finally:
            # Release our enriched table, but never an externally supplied table.
            if self._owned_vectors is not None:
                self._owned_vectors.drop_table_from_database_and_remove_from_cache()
                self._owned_vectors = None

    def _run_expectation_maximisation(
        self, cvv: SplinkDataFrame
    ) -> list[CoreModelSettings]:
        history, self._prior_history, self.converged = (
            expectation_maximisation_with_pairwise_prior(self, cvv)
        )
        return history

    def __repr__(self) -> str:
        rule = self._blocking_rule_for_training.blocking_rule_sql
        return f"<PairwisePriorTrainingSession, blocking on {rule}>"
