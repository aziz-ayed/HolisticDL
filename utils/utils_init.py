import argparse
import utils_model
from datetime import datetime

def define_parser():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--batch_range", type=int, nargs='+', default=[64], help="batch range")

	parser.add_argument("--network_size", type=int, nargs='+', default=[256, 128], help="size of network layers")

	parser.add_argument("--stab_ratio_range", type=float, nargs='+', default=[0.8], help="stability ratio range")

	parser.add_argument("--is_stable", action="store_true", help="stable version")

	parser.add_argument("--dropout", type=float, default=1, help="dropout rate, 1 is no dropout, 0 is all set to 0")

	parser.add_argument("--rho", "-r", type=float, default=0, help="Radius of the uncertainty set for robust training.")

	parser.add_argument("--robust_test", "-rtest", type=float,  nargs='+', default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], help="radius of the uncertainty set for robust testing.")

	parser.add_argument("--l2", type=float, default=0, help="l2 regularization parameter")

	parser.add_argument("--l0", type=float, default=0, help="l0 regularization parameter")

	parser.add_argument("--reg_stability", type=float, default=0, help="reg stability regularization parameter")

	parser.add_argument("--data_set", type=str, default="mnist", help="number of subsets")

	parser.add_argument("--train_size", type=float, default=0.80, help="percentage of data used of training")

	parser.add_argument("--lr", type=float, default=0.001, help="learning Rate used for the optimizer")

	parser.add_argument("--val_size", type=float, default=0.20, help="percentage of data used for validation")

	parser.add_argument("--n_distillations", type=int, default=1, help="nb of rounds of self-distillation for the NN; 1 means no self-distillation")

	return parser


def read_config_train(config):
	seed = config['random_seed']
	max_train_steps = config['max_num_training_steps']
	num_output_steps = config['num_output_steps']
	num_summ_steps = config['num_summary_steps']
	num_check_steps = config['num_checkpoint_steps']

	return seed, max_train_steps, num_output_steps, num_summ_steps, num_check_steps

def read_config_network(config, model):
	network_vars_w = [getattr(model, var) for var in  config['network_weights']]
	network_vars_b = [getattr(model, var) for var in  config['network_biases']]
	network_vars = network_vars_w + network_vars_b 
	sparsity_vars = [getattr(model, var) for var in  config['sparsity_variables']]
	stable_var = [getattr(model, config['stability_variable'])]

	return network_vars, sparsity_vars, stable_var

def read_train_args(args):
	rho = args.rho
	is_stable = args.is_stable
	learning_rate = args.lr
	l0 = args.l0
	l2 = args.l2
	batch_range = args.batch_range
	stab_ratio_range = args.stab_ratio_range
	dropout = args.dropout
	network_size = args.network_size

	return rho, is_stable, learning_rate, l0, l2, batch_range, stab_ratio_range, dropout, network_size

def read_data_args(args):
	data_set = args.data_set
	train_size = args.train_size
	val_size = args.val_size

	return data_set, train_size, val_size

def init_experiements(config, args, num_classes, num_features, data):
	num_experiments = config['num_experiments']
	dict_exp = utils_model.create_dict(args, num_classes, num_features, data.train.images.shape, data.test.images.shape)
	output_dir = 'outputs/logs/' + str(args.data_set) + '/' + str(datetime.now())
	return num_experiments, dict_exp, output_dir

