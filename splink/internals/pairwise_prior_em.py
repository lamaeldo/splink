"""EM with a frequency-dependent mixing proportion and native M/U updates."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from splink.internals.expectation_maximisation import (
    _max_change_in_parameters_comparison_levels,
    compute_new_parameters_sql,
    compute_proportions_for_new_parameters,
    maximisation_step,
)
from splink.internals.pairwise_prior import PairwisePriorParameters, _FrequencyPrior
from splink.internals.pipeline import CTEPipeline
from splink.internals.predict import _combine_prior_and_mws
from splink.internals.settings import CoreModelSettings, Settings

if TYPE_CHECKING:
    from splink.internals.pairwise_prior_training_session import (
        PairwisePriorTrainingSession,
    )
    from splink.internals.splink_dataframe import SplinkDataFrame

logger = logging.getLogger(__name__)
FREQUENCY_COLUMN = "__splink__prior_frequency"
PRIOR_GROUP = "__splink__prior_context"


def _prediction_sqls(
    session: PairwisePriorTrainingSession,
    core: CoreModelSettings,
    prior: _FrequencyPrior,
) -> list[dict[str, str]]:
    comparisons = core.comparisons
    gammas = [cc._gamma_column_name for cc in comparisons]
    if session.estimate_without_term_frequencies:
        columns = [*gammas, FREQUENCY_COLUMN, "agreement_pattern_count"]
        for cc in comparisons:
            cases = "\n".join(
                cl._match_weight_sql(cc._gamma_column_name)
                for cl in cc.comparison_levels
            )
            columns.append(f"CASE {cases} END as {cc._mw_column_name}")
        input_table = "__splink__agreement_pattern_counts"
        weight_columns = [cc._mw_column_name for cc in comparisons]
        retained_columns = [*gammas, FREQUENCY_COLUMN, "agreement_pattern_count"]
    else:
        columns = Settings.columns_to_select_for_match_weight_parts(
            unique_id_input_columns=session.unique_id_input_columns,
            comparisons=comparisons,
            retain_matching_columns=False,
            retain_intermediate_calculation_columns=False,
            additional_columns_to_retain=[],
        )
        columns.append(FREQUENCY_COLUMN)
        input_table = "__splink__df_comparison_vectors"
        weight_columns = [
            col for cc in comparisons for col in cc._match_weight_columns_to_sum
        ]
        retained_columns = [*gammas, FREQUENCY_COLUMN]

    # Native level weights, TF weights and stable probability SQL are shared
    # with ordinary prediction. Only the prior log odds differ.
    log_odds = (
        f"({prior.intercept!r}) + ({prior.coefficient!r}) * "
        f"(-ln(cast({FREQUENCY_COLUMN} as float8)) - ({prior.center!r}))"
    )
    match_weight, probability = _combine_prior_and_mws(
        0.5,
        [f"({log_odds}) / {math.log(2)!r}", *weight_columns],
        session.db_api.sql_dialect,
    )
    return [
        {
            "sql": f"select {', '.join(columns)} from {input_table}",
            "output_table_name": "__splink__df_match_weight_parts",
        },
        {
            "sql": (
                f"select {match_weight} as match_weight, "
                f"{probability} as match_probability, "
                f"{', '.join(retained_columns)} from __splink__df_match_weight_parts"
            ),
            "output_table_name": "__splink__df_predict",
        },
    ]


def expectation_maximisation_with_pairwise_prior(
    session: PairwisePriorTrainingSession,
    comparison_vectors: SplinkDataFrame,
) -> tuple[list[CoreModelSettings], list[PairwisePriorParameters], bool]:
    """Fit the logistic prior alongside M/U, without changing ordinary EM.

    Sufficient statistics are grouped by frequency as well as gamma values;
    grouping by gamma alone would erase the pairwise priors.
    The caller owns comparison_vectors; this function owns its grouped table.
    """
    core = session.core_model_settings
    history = [core.copy()]
    grouped = None
    vectors = comparison_vectors
    without_tf = session.estimate_without_term_frequencies
    gammas = [cc._gamma_column_name for cc in core.comparisons]
    columns = ", ".join([*gammas, FREQUENCY_COLUMN])
    try:
        if without_tf:
            pipeline = CTEPipeline([vectors])
            pipeline.enqueue_sql(
                f"select {columns}, count(*) as agreement_pattern_count "
                f"from {vectors.templated_name} group by {columns}",
                "__splink__agreement_pattern_counts",
            )
            grouped = session.db_api.sql_pipeline_to_splink_dataframe(pipeline)
            vectors = grouped

        weight = "agreement_pattern_count" if without_tf else "1"
        pipeline = CTEPipeline([vectors])
        pipeline.enqueue_sql(
            f"select {FREQUENCY_COLUMN} as frequency, sum({weight}) as n "
            f"from {vectors.templated_name} group by {FREQUENCY_COLUMN} "
            f"order by {FREQUENCY_COLUMN}",
            "__splink__prior_histogram",
        )
        histogram_df = session.db_api.sql_pipeline_to_splink_dataframe(pipeline)
        try:
            histogram = histogram_df.as_record_list()
        finally:
            histogram_df.drop_table_from_database_and_remove_from_cache()
        prior = _FrequencyPrior(
            [int(r["frequency"]) for r in histogram],
            [float(r["n"]) for r in histogram],
            session._prior_coefficient_bound,
        )
        prior.initialise(core.probability_two_random_records_match)
        prior_history = [prior.parameters(0)]
        if len(histogram) < 3:
            logger.warning(
                "Pairwise-prior training has only %d distinct frequency contexts. "
                "Constant frequency reduces to ordinary EM; two contexts may not "
                "identify the mixture. Check sensitivity to initial parameters.",
                len(histogram),
            )

        converged = False
        for iteration in range(1, session.training_settings.max_iterations + 1):
            pipeline = CTEPipeline([vectors])
            pipeline.enqueue_list_of_sqls(_prediction_sqls(session, core, prior))
            counts_sql = compute_new_parameters_sql(without_tf, core.comparisons)
            counts_sql += f"""
                union all
                select {FREQUENCY_COLUMN} as comparison_vector_value,
                    sum(match_probability * {weight}) as m_count,
                    sum((1-match_probability) * {weight}) as u_count,
                    '{PRIOR_GROUP}' as output_column_name
                from __splink__df_predict group by {FREQUENCY_COLUMN}
            """
            pipeline.enqueue_sql(counts_sql, "__splink__m_u_counts")
            params = session.db_api.sql_pipeline_to_splink_dataframe(pipeline)
            try:
                records = compute_proportions_for_new_parameters(params)
            finally:
                params.drop_table_from_database_and_remove_from_cache()

            context_records = {
                r["comparison_vector_value"]: r
                for r in records
                if r["output_column_name"] == PRIOR_GROUP
            }
            core = maximisation_step(
                session.training_fixed_probabilities,
                core,
                [r for r in records if r["output_column_name"] != PRIOR_GROUP],
            )
            mean = core.probability_two_random_records_match
            old_probabilities = prior.probabilities()
            # Native proportions normalise within each output column. Undo
            # that normalisation to recover E[match | frequency]. At a boundary
            # every context has the same responsibility.
            responsibilities = [
                min(max(context_records[f]["m_probability"] * mean / w, 0), 1)
                if 0 < mean < 1
                else mean
                for f, w in zip(prior.frequencies, prior.weights)
            ]
            prior.fit(mean, responsibilities)
            history.append(core)
            prior_history.append(prior.parameters(iteration))
            m_u_change = _max_change_in_parameters_comparison_levels(history)
            prior_change = max(
                abs(p - old) for p, old in zip(prior.probabilities(), old_probabilities)
            )
            change = max(m_u_change["max_abs_change_value"], prior_change)
            logger.info(
                "Iteration %d: maximum parameter/prior change %.6g; "
                "prior coefficient %.6g",
                iteration,
                change,
                prior.coefficient,
            )
            if change < session.training_settings.em_convergence:
                converged = True
                break

        if not converged:
            logger.warning(
                "Pairwise-prior EM reached max_iterations without convergence"
            )
        if prior_history[-1].at_coefficient_bound:
            logger.warning("Pairwise-prior coefficient reached its configured bound")
        return history, prior_history, converged
    finally:
        if grouped is not None:
            grouped.drop_table_from_database_and_remove_from_cache()
