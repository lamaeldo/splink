---
tags:
  - API
  - training
---
# Documentation for`EMTrainingSession`

`linker.training.estimate_parameters_using_expectation_maximisation` returns an object of type `EMTrainingSession` which has the following methods:

::: splink.internals.em_training_session.EMTrainingSession
    handler: python
    options:
      show_root_heading: false
      show_root_toc_entry: true
      show_source: false
      members_order: source

## Pairwise-prior training sessions

`linker.training.estimate_parameters_using_pairwise_prior` returns a
`PairwisePriorTrainingSession`. It inherits the EM history charts and adds
`prior_history` and `converged`. See the
[frequency-dependent training guide](../topic_guides/training/pairwise_prior.md)
for the model and its limitations.

::: splink.internals.pairwise_prior_training_session.PairwisePriorTrainingSession
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
        - prior_history

::: splink.internals.pairwise_prior.PairwisePriorParameters
    handler: python
    options:
      show_root_heading: true
      show_source: false




