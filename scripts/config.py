import argparse


def parse_arguments():
    """Read commandline arguments

    Returns:
        args (Namespace): input as well as default arguments
    """

    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument("--lags",  nargs="+", type=int, required=True) 
    parser.add_argument("--cv",  type=int, required=True) 
    parser.add_argument("--srm_k",  nargs="+", type=int, required=True) 
    parser.add_argument("--encoding_original_neural_data", action="store_true")
    parser.add_argument("--encoding_shared_space", action="store_true")
    parser.add_argument("--pca_regression", action="store_true")
    parser.add_argument("--srm_denoise", action="store_true")
    parser.add_argument("--srm_shared_space_generalization", action="store_true")
    parser.add_argument("--pca_generalisation_across_subject", action="store_true")
    parser.add_argument("--electrode_space_generalization", action="store_true")

    parser.add_argument("--srm_all_elec", action="store_true")
    parser.add_argument("--original_regression_all_elec", action="store_true")
    parser.add_argument("--syntactic_feature", action="store_true")
    parser.add_argument("--speech_feature", action="store_true")
    parser.add_argument("--different_layer", action="store_true")
    parser.add_argument("--layer_id",  nargs="+", type=int, required=False, default=0) 
    parser.add_argument("--different_size", action="store_true")
    parser.add_argument("--model_size",  type=str, required=False)



    args = parser.parse_args()
    return args