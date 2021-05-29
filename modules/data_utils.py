import nnlib.nnlib.data_utils.base


def load_data_from_arguments(args, build_loaders=True):
    data_selector = nnlib.nnlib.data_utils.base.DataSelector()
    return data_selector.parse(args, build_loaders=build_loaders)
