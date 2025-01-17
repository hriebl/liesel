"""
some tests for the engine
"""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from liesel.goose.builder import EngineBuilder
from liesel.goose.engine import (
    Engine,
    SamplingResult,
    _add_time_dimension,
    stack_for_multi,
)
from liesel.goose.epoch import EpochConfig, EpochState, EpochType
from liesel.goose.kernel_sequence import KernelSequence
from liesel.goose.models import DictModel
from liesel.goose.pytree import register_dataclass_as_pytree
from liesel.goose.types import Array, KeyArray, ModelInterface, ModelState

from .deterministic_kernels import DetCountingKernel, DetCountingKernelState


@register_dataclass_as_pytree
@dataclass
class FooQuant:
    error_code: int
    result: tuple[Array, Array]


class FooQauntGen:
    error_book: ClassVar[dict[int, str]] = {0: "no errors"}

    def __init__(self, identifier):
        self.identifier = identifier

    def set_model(self, model: ModelInterface):
        pass

    def has_model(self) -> bool:
        return False

    def generate(
        self, prng_key: KeyArray, model_state: ModelState, epoch: EpochState
    ) -> FooQuant:
        u = jax.random.normal(prng_key)
        return FooQuant(0, (u, model_state["x"]))


def test_add_time_dimension():
    tree = jax.jit(lambda x: x)((1.0, [1.0, {"f": jnp.array([1.0, 2.0])}]))

    def get_dims(t):
        return [t[0].shape, t[1][0].shape, t[1][1]["f"].shape]

    dims = [(), (), (2,)]
    assert dims == get_dims(tree)

    tree2 = _add_time_dimension(False, tree)
    dims2 = [
        (1,),
        (1,),
        (1, 2),
    ]
    assert dims2 == get_dims(tree2)

    tree0 = jax.jit(lambda x: x)(
        (jnp.array([1.0, 2.0]), [jnp.zeros((3, 3)), {"f": jnp.array([1.0, 2.0])}])
    )
    tree3 = _add_time_dimension(True, tree0)
    dims3 = [
        (2, 1),
        (3, 1, 3),
        (2, 1),
    ]
    assert dims3 == get_dims(tree3)


def t_test_engine():
    num_chains = 4
    epoch_configs = [
        EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
        EpochConfig(EpochType.FAST_ADAPTATION, 50, 1, None),
        EpochConfig(EpochType.BURNIN, 50, 1, None),
        EpochConfig(EpochType.POSTERIOR, 100, 1, None),
    ]

    ms = {"x": jnp.array(1), "y": jnp.array(-1)}
    mss = stack_for_multi([ms for _ in range(num_chains)])
    con = DictModel(lambda ms: -0.5 * ms["x"] ** 2 - 0.5 * ms["y"])
    ker0 = DetCountingKernel(["x"], DetCountingKernelState.default())
    ker1 = DetCountingKernel(["y"], DetCountingKernelState.default())
    ker0.set_model(con)
    ker1.set_model(con)

    ks = KernelSequence([ker0, ker1])

    seeds = jax.random.split(jax.random.PRNGKey(0), num_chains)

    engine = Engine(
        seeds,
        mss,
        ks,
        epoch_configs,
        25,
        con,
        ["x"],
        minimize_transition_infos=False,
        store_kernel_states=True,
        quantity_generators=[FooQauntGen("foo"), FooQauntGen("bar")],
    )

    engine.sample_all_epochs()

    results: SamplingResult = engine.get_results()

    print(results.positions.combine_all())
    print(results.transition_infos.combine_all())
    print(results.kernel_states.combine_all())

    print(results.get_posterior_samples())
    print(results.get_tuning_times())

    print(results.generated_quantities.unwrap().combine_all().unwrap())


def t_test_engine_builder() -> None:
    builder = EngineBuilder(seed=1, num_chains=4)

    builder.set_epochs(
        [
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.FAST_ADAPTATION, 50, 1, None),
            EpochConfig(EpochType.BURNIN, 50, 1, None),
            EpochConfig(EpochType.POSTERIOR, 100, 1, None),
        ]
    )
    ms = {"x": jnp.array(1), "y": jnp.array(-1)}
    builder.set_initial_values(ms, multiple_chains=False)
    con = DictModel(lambda ms: -0.5 * ms["x"] ** 2 - 0.5 * ms["y"])
    builder.set_model(con)
    builder.add_kernel(DetCountingKernel(["x"], DetCountingKernelState.default()))
    builder.add_kernel(DetCountingKernel(["y"], DetCountingKernelState.default()))
    builder.add_quantity_generator(FooQauntGen("foo"))
    builder.add_quantity_generator(FooQauntGen("bar"))
    builder.positions_excluded = ["y"]
    engine = builder.build()

    engine.sample_all_epochs()
    results: SamplingResult = engine.get_results()

    print(results.get_posterior_samples())
    print(results.get_tuning_times())

    print(results.generated_quantities.unwrap().combine_all().unwrap())
    print(results.transition_infos.combine_all().unwrap())


if __name__ == "__main__":
    t_test_engine_builder()
