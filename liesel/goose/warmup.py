"""
# Standard warmup schemes
"""

from functools import partial

from .epoch import EpochConfig, EpochType

_EpochConfig = partial(EpochConfig, thinning=1, optional=None)


def stan_epochs(
    warmup_duration: int = 1000,
    posterior_duration: int = 1000,
    init_duration: int = 75,
    term_duration: int = 50,
    base_duration: int = 25,
) -> list[EpochConfig]:
    """
    Sets up a list of `liesel.goose.epoch.EpochConfig`'s,
    following the Stan Development Team, [Stan Reference Manual (2021), Chapter 15.2](
    https://mc-stan.org/docs/2_28/reference-manual/hmc-algorithm-parameters.html).

    ## Parameters

    - `warmup_duration`: The number of warmup samples.
    - `posterior_duration`: The number of posterior samples.
    - `init_duration`: The number of samples in the *initial fast* adaptation epoch.
    - `term_duration`: The number of samples in the *final fast* adaptation epoch.
    - `base_duration`: The number of samples in the *first slow* adaptation epoch.
    """

    if warmup_duration < 20:
        raise ValueError("warmup_duration too short (< 20)")

    if warmup_duration < init_duration + term_duration + base_duration:
        raise ValueError(
            "warmup_duration too short "
            "(< init_duration + term_duration + base_duration)"
        )

    epochs = [_EpochConfig(EpochType.INITIAL_VALUES, duration=1)]
    epochs.append(_EpochConfig(EpochType.FAST_ADAPTATION, init_duration))

    time_left = warmup_duration - init_duration - term_duration
    this_time = base_duration

    while 3 * this_time <= time_left:
        epochs.append(_EpochConfig(EpochType.SLOW_ADAPTATION, this_time))
        time_left -= this_time
        this_time *= 2

    epochs.append(_EpochConfig(EpochType.SLOW_ADAPTATION, time_left))
    epochs.append(_EpochConfig(EpochType.FAST_ADAPTATION, term_duration))
    epochs.append(_EpochConfig(EpochType.POSTERIOR, posterior_duration))

    return epochs
