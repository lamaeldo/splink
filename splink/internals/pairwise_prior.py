"""Logistic mixing proportions for frequency-aware EM training.

The scalar lambda is the pair-weighted mean prior, not the logistic intercept.
For each candidate slope, solve the intercept's score equation exactly before
solving the profiled slope equation. This is a concave Bernoulli M-step.
No third-party numerical dependency or global monkey patch is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _expit(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


@dataclass(frozen=True)
class PairwisePriorParameters:
    """Parameters and range of the training prior at one EM iteration.

    logit(prior) = intercept + coefficient * (-log(frequency) - center).
    The mean is weighted by candidate pairs, rather than by distinct blocks.
    """

    iteration: int
    mean: float
    intercept: float
    coefficient: float
    center: float
    minimum: float
    maximum: float
    at_coefficient_bound: bool


class _FrequencyPrior:
    def __init__(
        self, frequencies: list[int], counts: list[float], coefficient_bound: float
    ):
        self.frequencies = frequencies
        total = sum(counts)
        self.weights = [count / total for count in counts]
        raw_z = [-math.log(frequency) for frequency in frequencies]
        self.center = math.fsum(w * z for w, z in zip(self.weights, raw_z))
        self.z = [z - self.center for z in raw_z]
        self.coefficient_bound = coefficient_bound
        self.coefficient = 0.0
        self.intercept = 0.0

    def probabilities(self) -> list[float]:
        return [_expit(self.intercept + self.coefficient * z) for z in self.z]

    def _intercept(self, target: float, coefficient: float) -> float:
        if not 0 < target < 1:
            raise ValueError(
                "Pairwise-prior EM requires a finite mean strictly between 0 and 1. "
                "A boundary mean indicates a collapsed mixture; check the initial "
                "parameters and the information available after blocking."
            )
        # Calibrate the smaller component. Near one, summing probabilities would
        # discard relative accuracy in the nonmatch tail through cancellation.
        direction = 1 if target <= 0.5 else -1
        tail_target = target if direction == 1 else 1 - target
        log_odds = math.log(tail_target) - math.log1p(-tail_target)
        offsets = [direction * coefficient * z for z in self.z]
        lower = log_odds - max(offsets) - 1
        upper = log_odds - min(offsets) + 1
        intercept = min(max(log_odds, lower), upper)
        tolerance = max(1e-12 * tail_target, math.ulp(tail_target))
        for _ in range(200):
            probabilities = [_expit(intercept + offset) for offset in offsets]
            error = (
                math.fsum(w * p for w, p in zip(self.weights, probabilities))
                - tail_target
            )
            if abs(error) <= tolerance:
                return direction * intercept
            if error > 0:
                upper = intercept
            else:
                lower = intercept
            derivative = math.fsum(
                w * p * (1 - p) for w, p in zip(self.weights, probabilities)
            )
            candidate = intercept - error / derivative if derivative > 0 else math.inf
            intercept = candidate if lower < candidate < upper else (lower + upper) / 2
        raise ArithmeticError("Could not calibrate the pairwise-prior intercept")

    def initialise(self, mean: float) -> None:
        self.intercept = self._intercept(mean, 0)

    def fit(self, mean: float, responsibilities: list[float]) -> None:
        if len(self.z) == 1:
            self.coefficient = 0.0
            self.intercept = self._intercept(mean, 0)
            return

        use_nonmatch_tail = mean > 0.5
        observed = math.fsum(
            w * (1 - r if use_nonmatch_tail else r) * z
            for w, r, z in zip(self.weights, responsibilities, self.z)
        )

        def gradient(coefficient: float) -> tuple[float, float]:
            intercept = self._intercept(mean, coefficient)
            fitted = math.fsum(
                w
                * z
                * _expit(
                    -(intercept + coefficient * z)
                    if use_nonmatch_tail
                    else intercept + coefficient * z
                )
                for w, z in zip(self.weights, self.z)
            )
            return (
                intercept,
                fitted - observed if use_nonmatch_tail else observed - fitted,
            )

        lower, upper = -self.coefficient_bound, self.coefficient_bound
        if gradient(lower)[1] <= 0:
            coefficient = lower
        elif gradient(upper)[1] >= 0:
            coefficient = upper
        else:
            # The profiled score is monotone decreasing, even when its root is
            # negative. Bisection also handles nearly separated data reliably.
            for _ in range(50):
                coefficient = (lower + upper) / 2
                if gradient(coefficient)[1] > 0:
                    lower = coefficient
                else:
                    upper = coefficient
                if upper - lower < 1e-10:
                    break
            coefficient = (lower + upper) / 2
        self.coefficient = coefficient
        self.intercept = self._intercept(mean, coefficient)

    def parameters(self, iteration: int) -> PairwisePriorParameters:
        probabilities = self.probabilities()
        return PairwisePriorParameters(
            iteration=iteration,
            mean=math.fsum(w * p for w, p in zip(self.weights, probabilities)),
            intercept=self.intercept,
            coefficient=self.coefficient,
            center=self.center,
            minimum=min(probabilities),
            maximum=max(probabilities),
            at_coefficient_bound=abs(self.coefficient) == self.coefficient_bound,
        )
