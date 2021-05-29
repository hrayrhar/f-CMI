from nnlib.nnlib import nn_utils as nn_utils_base


def parse_network_from_config(args, input_shape, detailed_output=False):
    # parse project-specific networks
    # none in this case

    # parse general-case networks
    net, output_shape = nn_utils_base.parse_network_from_config(args, input_shape)

    if detailed_output:
        net = nn_utils_base.StandardNetworkWrapper(net, output_name='pred')

    return net, output_shape
