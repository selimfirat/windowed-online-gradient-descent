from experiment import Experiment
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Experimentation Runner")
    parser.add_argument("--model", default="one_step_rnn", type=str)
    parser.add_argument("--recurrent_cell", default="rnn", choices=["rnn", "lstm", "gru"], type=str)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "windowed_ogd", "adam", "rmsprop"], type=str)
    parser.add_argument("--data", default="elevator_1000", choices=["elevator", "elevator_1000", "puma32f", "pumadyn", "alcoa", "euro"], type=str)
    parser.add_argument("--hidden_size", default=50, type=int)
    parser.add_argument("--window_size", default=25, type=int)
    parser.add_argument("--alpha", default=0, type=int)
    parser.add_argument("--clip_hh", default=0.95, type=float)
    parser.add_argument("--clip_ih", default=0.95, type=float)
    parser.add_argument("--time_decay", default=False, action="store_true", help="Whether to multiply the gradient of ")
    parser.add_argument("--truncation", default=30, type=int)
    parser.add_argument("--mlflow", default=False, action="store_true")
    parser.add_argument("--experiment_name", default="wogd_experiments", type=str)
    parser.add_argument("--lr_ih", default=0.001, type=float)
    parser.add_argument("--lr_hh", default=0.001, type=float)
    parser.add_argument("--lr_oh", default=0.001, type=float)
    parser.add_argument("--output_decay", default=1.0, type=float) # for windowed_ogd only
    parser.add_argument("--time_decay_power", default=0.5, type=float)
    parser.add_argument("--log_hessian", default=False, action="store_true")
    parser.add_argument("--log_differences", default=False, action="store_true")
    parser.add_argument("--append_input", default=False, action="store_true")
    parser.add_argument("--log_hessian_every", default=50, type=int)
    parser.add_argument("--num_threads", default=1, type=int)

    args = vars(parser.parse_args())

    experiment = Experiment(**args)

    experiment.run()
