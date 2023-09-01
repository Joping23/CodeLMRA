import argparse


def args_to_string(args):
    """
    Transform experiment's arguments into a string
    :param args:
    :return: string
    """
    args_string = ""

    args_to_show = ["experiment", "bz",
                    "lr", "lr_scheduler", "optimizer", "fit_by_epoch", "num_local_steps", "precentage_attack", "DP", "epsilon"]

    for arg in args_to_show:
        args_string += arg
        args_string += "_" + str(getattr(args, arg)) + "_"

    return args_string[:-1]


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment',
        help='name of experiment',
        type=str)
    parser.add_argument(
        "--use_weighted_average",
        help="if used the weighted average will be optimized, otherwise the average is optimized,"
             " i,e, all the local functions are treated the same.",
        action='store_true'
    )
    parser.add_argument(
        '--model',
        help='model type: linear, neural',
        type=str,
        default="linear"
    )
    parser.add_argument(
        '--fit_by_epoch',
        help='if chosen each local step corresponds to one epoch,'
             ' otherwise each local step corresponds to one gradient step',
        action='store_true'
    )
    parser.add_argument(
        '--num_workers',
        help='number of workers;',
        type=int,
        default=2
    )
    parser.add_argument(
        '--runs',
        help='number of runs for the experiments;',
        type=int,
        default=1
    )
    parser.add_argument(
        '--num_rounds',
        help='number of communication rounds;',
        type=int,
        default=1
    )
    parser.add_argument(
        '--bz',
        help='batch_size;',
        type=int,
        default=1
    )
    parser.add_argument(
        '--num_local_steps',
        help='number of local steps before communication;',
        type=int,
        default=1
    )
    parser.add_argument(
        '--log_freq',
        help='number of local steps before communication;',
        type=int,
        default=1
    )
    parser.add_argument(
        '--device',
        help='device to use, either cpu or gpu;',
        type=str,
        default="cpu"
    )
    parser.add_argument(
        '--optimizer',
        help='optimizer to be used for the training;',
        type=str,
        default="sgd"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help='learning rate',
        default=1e-2
    )
    parser.add_argument(
        "--adv_lr",
        type=float,
        help='adversary learning rate',
        default=1e-3
    )
    parser.add_argument(
        "--lr_scheduler",
        help='learning rate decay scheme to be used;'
             ' possible are "cyclic", "sqrt", "linear" and "constant"(no learning rate decay);'
             'default is "cyclic"',
        type=str,
        default="constant"
    )
    parser.add_argument(
        "--adversary_ability",
        help='adversary ability to interrupt the FL process:'
             'none: adversary can just listen for the models'
             'personalized_attack: adversary track the personalized model in the end of trainining'
             'randomized_attack: adversary randomly give inputs to the training'
             'intermediate_attack: intermediate adversary randomly give inputs to the training and give the wrong answer back to the server',
        type=str,
        default="none"
    )
    parser.add_argument(
        "--precentage_attack",
        type=float,
        help='the precentage of the iterations which allows the adversary to corrupt the inputs',
        default=0.1
    )
    parser.add_argument(
        "--sigma",
        help='noise variance',
        type=float,
        default="0.1"
    )
    parser.add_argument(
        "--gnetwork_type",
        help='Gradient network structure'
             'nn_linear: neural network with one hidden fully-connected layer'
             'nn_multiple_linear: two hidden fully-connected layer',
        type=str,
        default="nn_linear"
    )
    parser.add_argument(
        "--gnetwork_features",
        help='the number of features in the Gradient network',
        type=int,
        default="1000"
    )
    parser.add_argument(
        "--gnetwork_num_epochs",
        help='the number of epochs to train the Gradient network',
        type=int,
        default="20000"
    )
    parser.add_argument(
        "--decoded_epochs",
        help='the number of epochs to decode the optimum',
        type=int,
        default="20000"
    )
    parser.add_argument(
        "--start_point",
        help='The start point for getting the local optimum from the gradient network'
            'global_model'
            'random'
            'zeros',
        type=str,
        default="global_model"
    )

    parser.add_argument(
        "--num_trials_to_decode",
        help='the number of trails to decode for each worker',
        type=int,
        default="20"
    )

    parser.add_argument(
        "--DP",
        help='to enable differential privacy in training',
        default=False,
        action='store_true'

    )
    parser.add_argument(
        "--epsilon",
        help='target epsilon to be used in the privacy engine',
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--max_grad_norm",
        help='the clipping threshold of the gradients, to be used in the privacy engine',
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--early_stop",
        help='to enable early stop',
        default=False,
        action='store_true'

    )
    parser.add_argument(
        "--patience",
        help='patience level for early stopping of training gradient network',
        type=int,
        default=300
    )
    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args
