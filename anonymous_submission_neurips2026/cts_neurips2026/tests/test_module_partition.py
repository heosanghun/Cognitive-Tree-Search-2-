from cts.model.module_partition import LAYER_TO_MODULE, layers_for_module


def test_forty_two_layers_nineteen_modules():
    assert len(LAYER_TO_MODULE) == 42
    assert max(LAYER_TO_MODULE.values()) == 18
    assert min(LAYER_TO_MODULE.values()) == 0
    assert len(layers_for_module(0)) >= 1
