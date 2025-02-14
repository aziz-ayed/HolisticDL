from datetime import datetime
import numpy as np
import csv
from utils import total_gini
import tensorflow.compat.v1 as tf
import json
from pgd_attack import LinfPGDAttack
from utils_MLP_model import init_MLP_vars

with open('config.json') as config_file:
    config = json.load(config_file)

w_vars, b_vars, stable_var, sparse_vars = init_MLP_vars()

def print_metrics(sess, model, train_dict, nat_dict, val_dict, val_dict_distil, test_dict, ii, args, summary_writer, dict_exp, experiment, global_step):
    print('Step {}:    ({})'.format(ii, datetime.now()))
    try:
        nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
        print('    batch training nat accuracy {:.4}'.format(nat_acc * 100))
        nat_xent = sess.run(model.xent, feed_dict=nat_dict)
        print('    Nat Xent {:.4}'.format(nat_xent))

        stable_xent = sess.run(model.stable_xent, feed_dict=nat_dict)
        robust_xent = sess.run(model.robust_xent, feed_dict=nat_dict)
        robust_stable_xent = sess.run(model.robust_stable_xent, feed_dict=nat_dict)

        train_l2 = sess.run(model.l2_loss, feed_dict=nat_dict)
        print('    Batch Training L2 Loss {:.4}'.format(train_l2))

    except:
        train_distil_loss = sess.run(model.distil_loss, feed_dict=nat_dict)
        print('    Batch Training Distillation L2 Teacher Student Loss {:.4}'.format(train_distil_loss))

    train_normal_acc = sess.run(model.accuracy, feed_dict=train_dict)
    print('    Training accuracy {:.4}'.format(train_normal_acc * 100))
    train_l2 = sess.run(model.l2_loss, feed_dict=train_dict)
    print('    Training L2 Loss Ground Truth {:.4}'.format(train_l2))
    summary3 = tf.Summary(value=[tf.Summary.Value(tag='TrainL2', simple_value=train_l2), ])
    val_l2 = sess.run(model.l2_loss, feed_dict=val_dict)
    print('    Val L2 Loss Ground Truth {:.4}'.format(val_l2))
    test_l2 = sess.run(model.l2_loss, feed_dict=test_dict)
    print('    Test L2 Loss Ground Truth {:.4}'.format(test_l2))
    val_acc = sess.run(model.accuracy, feed_dict=val_dict)
    print('    validation nat accuracy {:.4}'.format(val_acc * 100))

    if args.n_distillations > 1:
        train_l2 = sess.run(model.distil_loss, feed_dict=nat_dict)
        print('    Training L2 Loss vs Teacher {:.4}'.format(train_l2))
        val_distil_loss = sess.run(model.distil_loss, feed_dict=val_dict_distil)
        print('    Validation L2 Loss Ground Truth {:.4}'.format(val_distil_loss))
        summary7 = tf.Summary(value=[tf.Summary.Value(tag='ValTeacherL2', simple_value=val_distil_loss), ])
        summary_writer.add_summary(summary7, global_step.eval(sess))

    # summary1 = tf.Summary(value=[tf.Summary.Value(tag='TrainAcc', simple_value=train_normal_acc),])
    summary2 = tf.Summary(value=[tf.Summary.Value(tag='ValAcc', simple_value=val_acc),])
    summary4 = tf.Summary(value=[tf.Summary.Value(tag='ValL2', simple_value=val_l2), ])
    summary6 = tf.Summary(value=[tf.Summary.Value(tag='TrainTeacherL2', simple_value=train_l2), ])
    # summary_writer.add_summary(summary1, global_step.eval(sess))
    summary_writer.add_summary(summary2, global_step.eval(sess))
    # summary_writer.add_summary(summary3, global_step.eval(sess))
    summary_writer.add_summary(summary4, global_step.eval(sess))
    summary_writer.add_summary(summary6, global_step.eval(sess))
    #summary_writer.add_text('args', str(args), global_step.eval(sess))
    # summary5 = sess.run(model.summary, feed_dict=test_dict)
    # summary_writer.add_summary(summary5, global_step.eval(sess))
    test_acc = sess.run(model.accuracy, feed_dict=test_dict)
    print('    Test accuracy {:.4}'.format(test_acc * 100))
    # summary_writer.add_summary(test_acc, global_step.eval(sess))

    if args.is_stable:
        stable_var = sess.run(getattr(model, config['stability_variable']), feed_dict=nat_dict)
        print('    Stability Variable {:.4}'.format(stable_var ))
        print('    Stable Xent {:.4}'.format(stable_xent))
        print('    Stable Xent {:.4}'.format(stable_xent))

    if args.rho > 0 :
        print('    Robust Xent {:.4}'.format(robust_xent))
        if args.is_stable:
            print('    Robust Stable Xent {:.4}'.format(robust_stable_xent))

    for i in range(len(w_vars)):
        if args.l0 > 0:
            print('    Killed neurons - ' + w_vars[i], dict_exp[w_vars[i] + '_killed_neurons'][experiment])
            print('    Killed input neurons - ' + w_vars[i], dict_exp[w_vars[i] + '_killed_input_features'][experiment])

        print('    Non zero features percentage - ' + w_vars[i] , dict_exp[w_vars[i] + '_nonzero'][experiment])

    regularizer = sess.run(model.regularizer, feed_dict=nat_dict)
    print('    Regularizer', regularizer)


    # try:
    # summary = tf.Summary(value=[
    #     tf.Summary.Value(tag='Train Xent', simple_value= nat_xent),
    #     # tf.Summary.Value(tag='Val Acc', simple_value= val_acc),
    #     tf.Summary.Value(tag='Train Acc', simple_value= nat_acc),
    #     tf.Summary.Value(tag='Train Stable Xent', simple_value= stable_xent),
    #     tf.Summary.Value(tag='Train Robust Stable Xent', simple_value= robust_stable_xent),
    #     tf.Summary.Value(tag='Test Acc', simple_value= test_acc)])
    # except:
        # pass

    for i in range(len(w_vars)):
        if args.l0 > 0:
            summary_sparse = tf.Summary(value=[
                tf.Summary.Value(tag=w_vars[i] + '_killed_neurons', simple_value=dict_exp[w_vars[i] + '_killed_neurons'][experiment]),
                tf.Summary.Value(tag=w_vars[i] + '_killed_inputs', simple_value=dict_exp[w_vars[i] + '_killed_input_features'][experiment]),
                tf.Summary.Value(tag=w_vars[i] + '_nonzero', simple_value=dict_exp[w_vars[i] + '_nonzero'][experiment])])

            summary_writer.add_summary(summary_sparse, global_step.eval(sess))


def update_dict_output(dict_exp, experiment, sess, test_acc, model, test_dict, num_iters):
    dict_exp['test_accs'][experiment] = test_acc*100
    dict_exp['iterations'][experiment] = num_iters

    return dict_exp

def update_adv_acc(args, best_model, x_test, y_test, experiment, dict_exp):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        clip = True
        if "uci" in args.data_set:
            clip = False

        for rho_test in args.robust_test:
            attack = LinfPGDAttack(best_model, rho_test, config['k'], config['a'], 
            config['random_start'], config['loss_func'], clip)
            x_test_adv = attack.perturb(x_test, y_test, sess)
            adv_dict = {best_model.x_input: x_test_adv, best_model.y_input: y_test}
            dict_exp['adv_test_accs'][rho_test][experiment] = sess.run(best_model.accuracy, feed_dict=adv_dict)


def print_stability_measures(dict_exp, args, num_experiments, batch_size, subset_ratio, tot_test_acc, tot_train_acc, max_train_steps, network_path):

    avg_test_acc = tot_test_acc / num_experiments
    avg_train_acc = tot_train_acc / num_experiments
    std = np.array([float(k) for k in dict_exp['test_accs']]).std()
    logit_stability = np.mean(np.std(dict_exp['logits_acc'], axis=0), axis=0)
    gini_stability = total_gini(dict_exp['preds'].transpose())

    print('  Average training accuracy {:.4}'.format(avg_train_acc * 100))
    print('  Average testing accuracy {:.4}'.format(avg_test_acc * 100))
    print('  Individual accuracies: \n', dict_exp['test_accs'])
    print('  Adv testing accuracies', dict_exp['adv_test_accs'])
    print('  Stability values', dict_exp[stable_var])
    print('  Test Accuracy std {:.2}'.format(np.array([float(k) for k in dict_exp['test_accs']]).std()))
    print("  Logits std", np.mean(np.mean(np.std(dict_exp['logits_acc'], axis=0), axis=0)))
    print("  Gini stability", gini_stability)


    weights_stability = print_layer_stability_ff(dict_exp, num_experiments)
    weights_nonzero = [np.mean(dict_exp[w_vars[i]]) for i in range(len(w_vars))]

    for i in range(len(w_vars)):
        print(w_vars[i] + ' non zero percentage', weights_nonzero[i])
    

    file = open(str('results_' + network_path + args.data_set + '.csv'), 'a+', newline='')

    file_read = open(str('results_' + network_path + args.data_set + '.csv'), "r")
    one_char = file_read.read(1)

    writer = csv.writer(file)

    if not len(one_char):
        headers = []
        headers += ['num_experiments', 'batch_size', 'subset_ratio', 'max_train_steps']
        headers += ['test accuracy '+ str(i) for i in range(num_experiments)]

        for key in dict_exp:
            if key not in w_vars+ b_vars+ [stable_var]+ sparse_vars + ['adv_test_accs', 'preds']:
                headers +=  ['Avg '+str(key)]

        headers += ['Avg test adversarial acc for rho = '+ str(rho) for rho in  args.robust_test]
        headers += ['is_stable', 'rho', 'train_size', 'l2', 'l0', 'network_size', 'learning rate']
        headers += [w_vars[i] + ' Nonzero weights' for i in range(len(w_vars))]
        headers += [w_vars[i] + ' Stability' for i in range(len(w_vars))]
        headers += ['std', 'logit_stability', 'gini_stability' ] 
        writer.writerow(headers)

    with file:

        cols = []

        cols += [num_experiments, batch_size, subset_ratio, max_train_steps]
        cols += [dict_exp['test_accs'][i] for i in range(num_experiments)]
        
        for key in dict_exp:
            if key not in w_vars+ b_vars+ [stable_var]+ sparse_vars + ['adv_test_accs', 'preds']:
                cols += [np.mean(dict_exp[key])]

        cols += [np.mean(dict_exp['adv_test_accs'][rho]) for rho in args.robust_test]
        cols += [args.is_stable, args.rho,  args.train_size, args.l2, args.l0, args.network_size, args.lr]
        cols += weights_nonzero
        cols += weights_stability
        cols += [std, logit_stability, gini_stability ] 

        print(cols)





        writer.writerow(cols)


def print_layer_stability_ff(dict_exp, num_experiments):

    stabilities = []
    for i in range(len(w_vars)):
        w_i = [dict_exp[w_vars[i]][experiment].reshape(-1) for experiment in range(num_experiments)]
        w_stability = np.mean(np.std(w_i, axis=0), axis=0)
        print(w_vars[i] + " std", w_stability)
        stabilities = stabilities + [w_stability]
    return stabilities
