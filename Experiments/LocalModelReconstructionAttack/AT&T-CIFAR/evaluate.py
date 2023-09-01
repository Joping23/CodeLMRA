import json
from utils.args import parse_args
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt


def performance_multi_runs(args, epsilons, df,  accu="extraction_accuracy"):
    for e in epsilons:
        if e <= 2000:
            args.epsilon = e
            args.DP = True
        else:
            args.epsilon = 1.0
            args.DP = False

        file_name = args.experiment + "_w_" + str(args.num_workers) + "_lr_" + str(
            args.lr) + "_bz_" + str(args.bz) + "_fit_epoch_" + str(args.fit_by_epoch) + "_local_step_" + str(
            args.num_local_steps) \
                         + "_start_point_" + args.start_point + "_ability_" + args.adversary_ability + "_pre_" + str(
            args.precentage_attack) + "_dp_" + str(args.DP) + "_epsilon_" + str(args.epsilon) + ".json"

        run = 0
        with open(file_name, 'r') as f:
            data = json.load(f)
            for d in data:
                if run == 0 : print(d['args'])
                run += 1
                for i in range(args.num_workers):
                    if args.DP == False:
                        print(d["gradient_networks_loss"][0][str(i)])
                        #print(d['decoded_model'][str(i)]['grad_norm'])
                        df.loc[len(df.index)] = [2500, i, run, d['decoded_model'][str(i)][accu],
                                                 d['server_local_model'][str(i)][accu], (d['decoded_model'][str(i)][accu]- d['server_model'][accu][i])/(1-d['server_model'][accu][i]),
                                                 d['gradient_networks_loss'][1][str(i)],d['decoded_model'][str(i)]['grad_norm'],
                                                 sum(d["gradient_networks_norm"][str(i)])/len(d["gradient_networks_norm"][str(i)]),
                                                 sum(d["output_y_norm"][str(i)])/len(d["output_y_norm"][str(i)]),
                                                 torch.var(torch.tensor(d["gradient_networks_norm"][str(i)])).item(),
                                                 torch.var(torch.tensor(d["output_y_norm"][str(i)])).item()
                                                 ]
                    else:
                        df.loc[len(df.index)] = [args.epsilon, i, run, d['decoded_model'][str(i)][accu], d['server_model'][accu][i],
                                                 (d['decoded_model'][str(i)][accu]-d['server_model'][accu][i])/(1-d['server_model'][accu][i]),
                                                 d['gradient_networks_loss'][1][str(i)]/args.num_rounds,
                                                 d['decoded_model'][str(i)]['grad_norm'],
                                                 sum(d["gradient_networks_norm"][str(i)]) / len(d["gradient_networks_norm"][str(i)]),
                                                 sum(d["output_y_norm"][str(i)])/len(d["output_y_norm"][str(i)]),
                                                 torch.var(torch.tensor(d["gradient_networks_norm"][str(i)])).item(),
                                                 torch.var(torch.tensor(d["output_y_norm"][str(i)])).item()
                                                 ]

def performance_multi_runs_features(args, features, df, accu="extraction_accuracy"):
    args.DP = False
    for fea in features:
        file_name = args.experiment + "_w_" + str(args.num_workers) + "_lr_" + str(
            args.lr) + "_bz_" + str(args.bz) + "_fit_epoch_" + str(args.fit_by_epoch) + "_local_step_" + str(
            args.num_local_steps) \
                         + "_start_point_" + args.start_point + "_ability_" + args.adversary_ability + "_pre_" + str(
            args.precentage_attack) + "_dp_" + str(args.DP) + "_epsilon_" + str(args.epsilon) + "_feature_" + str(fea) + ".json"

        run = 0
        with open(file_name, 'r') as f:
            data = json.load(f)
            for d in data:
                run += 1
                for i in range(args.num_workers):
                    df.loc[len(df.index)] = [fea, i, run, d['decoded_model'][str(i)][accu], d['server_local_model'][str(i)][accu]]

def performance_multi_runs_localsteps(args, localsteps, df, accu="extraction_accuracy"):
    args.DP = False
    for ls in localsteps:
        file_name = args.experiment + "_w_" + str(args.num_workers) + "_lr_" + str(
            args.lr) + "_bz_" + str(args.bz) + "_fit_epoch_" + str(args.fit_by_epoch) + "_local_step_" + str(
            ls) \
                         + "_start_point_" + args.start_point + "_ability_" + args.adversary_ability + "_pre_" + str(
            args.precentage_attack) + "_dp_" + str(args.DP) + "_epsilon_" + str(args.epsilon) + ".json"

        run = 0
        with open(file_name, 'r') as f:
            data = json.load(f)
            for d in data:
                run += 1
                for i in range(args.num_workers):
                    df.loc[len(df.index)] = [ls, i, run, d['decoded_model'][str(i)][accu], d['server_local_model'][str(i)][accu],
                                             (d['decoded_model'][str(i)][accu]- d['server_model'][accu][i])/(1-d['server_model'][accu][i])]
    return d['local_model_accuracy_FL']

def performance_multi_runs_batches(args, batches, df, accu="extraction_accuracy"):
    args.DP = False
    for bz in batches:
        file_name = args.experiment + "_w_" + str(args.num_workers) + "_lr_" + str(
            args.lr) + "_bz_" + str(bz) + "_fit_epoch_" + str(args.fit_by_epoch) + "_local_step_1" \
                    + "_start_point_" + args.start_point + "_ability_" + args.adversary_ability + "_pre_" \
                    + str(args.precentage_attack) + "_dp_" + str(args.DP) + "_epsilon_" + str(args.epsilon) + ".json"

        run = 0
        with open(file_name, 'r') as f:
            data = json.load(f)
            for d in data:
                run += 1
                for i in range(args.num_workers):
                    #print(d.keys())
                    #print(d['gradient_networks_loss'])
                    #print(d['decoded_model'][str(i)]['grad_norm'])
                    df.loc[len(df.index)] = [bz, i, run, d['decoded_model'][str(i)][accu], d['server_local_model'][str(i)][accu],
                                             0,
                                             d['gradient_networks_loss'][1][str(i)], d['decoded_model'][str(i)]['grad_norm']]

def print_epsilons_vs_acc(df, clients = range(5), accu="extraction_accuracy"):
    colors = ['b', 'r', 'k', 'c', 'm']
    colors_chosen = [colors[i] for i in clients]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for client, color in zip(clients, colors_chosen):
        df_to_print = df[df['client'] == client]
        print(sum(df_to_print["Server_accu"])/len(df_to_print["Server_accu"]))
        print(sum(df_to_print["Decode_accu"])/len(df_to_print["Decode_accu"]))
        print("------------------------")
        sns.lineplot(data=df_to_print, x="epsilon", y="Decode_accu", markers=True, dashes=True, color=color, label=f"Client {client}'s decoded model" )
        sns.lineplot(data=df_to_print, x="epsilon", y="Server_accu", markers=True, dashes=False, color=color, label=f"Client {client}'s server model")
    fig.canvas.draw()
    for i in range(len(clients)*2):
        if i%2==0:
            ax.lines[i].set_linestyle("--")

    labels = [item.get_text() for item in ax.get_xticklabels()]
    print(labels)
    plt.legend()
    """
    labels[-2] = 'inf'
    ax.set_xticklabels(labels)
    """
    plt.xscale("log",basex=2)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(accu)
    plt.show()

def print_epsilons_vs_gap(df, clients = range(5), accu="extraction_accuracy"):
    colors = ['b', 'r', 'k', 'c', 'm']
    colors_chosen = [colors[i] for i in clients]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for client, color in zip(clients, colors_chosen):
        df_to_print = df[df['client'] == client]
        sns.lineplot(data=df_to_print, x="epsilon", y="Gap", label=f"Client {client}", color=color)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    plt.legend()
    labels[-2] = 'inf'
    ax.set_xticklabels(labels)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(f"Gap "+accu)
    plt.show()

def print_features_vs_acc(df, property = "feature_number", accu="extraction_accuracy"):
    clients = range(3)
    colors = ['b', 'r', 'k', 'c', 'm', 'c', 'c', 'c', 'c', 'c']
    colors_chosen = [colors[i] for i in clients]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for client, color in zip(clients, colors_chosen):
        df_to_print = df[df['client'] == client]
        print(df_to_print["Decode_accu"], df_to_print["Server_accu"])
        print(sum(df_to_print["Decode_accu"])/len(df_to_print["Decode_accu"]))
        print(sum(df_to_print["Server_accu"])/len(df_to_print["Server_accu"]))
        print("------------------------")
        sns.lineplot(data=df_to_print, x=property, y="Server_accu",linestyle='--', color=color,  label=rf"$\tilde \theta_{client}$")
        sns.lineplot(data=df_to_print, x=property, y="Decode_accu", color=color,  label=rf"$\hat\theta^*_{client}$")
        #sns.lineplot(data=df_to_print, x=property, y="Gap", color=color,
        #             label=f"Client {client}'s decoded model")

    fig.canvas.draw()
    for i in range(len(clients)*2):
        if i%2==0:
            ax.lines[i].set_linestyle("--")
    plt.xscale("log", basex=2)
    #plt.xlabel(property)
    #plt.ylabel(accu)
    plt.xlabel("Batch Size", size=12)
    plt.ylabel("Test accuracy", size=12)
    plt.legend(ncol=3)
    plt.show()

def print_features_vs_chosen_y(df, x = "batches", y="Gradient_network_loss"):
    clients = range(3)
    colors = ['b', 'r', 'k', 'c', 'm', "c", "c", "c", "c", "c"]
    colors_chosen = [colors[i] for i in clients]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for client, color in zip(clients, colors_chosen):
        df_to_print = df[df['client'] == client]
        print(df_to_print[y])
        sns.lineplot(data=df_to_print, x=x, y=y,color=color,  label=f"Client {client}")
    fig.canvas.draw()
    plt.xscale("log", basex=2)
    #plt.xlabel(x)
    #plt.ylabel(y)
    plt.xlabel("Batch size", size=12)
    plt.ylabel("Mapping loss", size=12)
    plt.show()

def print_FL_acc_vs_iter(acc, df):
    colors = ['b', 'r', 'k', 'c', 'm']
    for worker in range(len(acc)):
        server_accu = df[df["client"] == worker]['Server_accu']
        decode_accu = df[df["client"] == worker]['Decode_accu']
        train_acc = acc[f'{worker}']
        print(decode_accu)
        length = len(train_acc[::])
        plt.plot(range(length), [decode_accu for i in range(length)], '--', color=colors[worker])
        plt.plot(range(length), [server_accu for i in range(length)], '-', color=colors[worker])
        plt.plot(range(length), train_acc[::], label=f'Client {worker}', color=colors[worker])
    plt.legend()
    plt.title("Local step 10")
    plt.ylabel("Train accuracy")
    plt.xlabel("Iteration")
    plt.show()

def show_epsilons_effect(args, chosen_clients=range(5), accu="extraction_accuracy"):
    #epsilons = [10.0,40.0,80.0,160.0,200]
    epsilons = [2500]
    #epsilons = [10.0,100.0,200.0,300.0,1000.0]
    #epsilons = [1400]
    #epsilons = [0.5,1.0,5.0,15.0, 20.0, 40.0,80.0,160.0,500.0,1000.0,2000.0,2500 ]
    d = {"epsilon": [], "client": [], "run": [], "Decode_accu": [], "Server_accu": [], "Gap":[],
         "Gradient_network_loss":[], "Gradient_norm":[], "Gradient_network_prediction_mean":[], "output_y_mean":[],
         "Gradient_network_prediction_var":[], "output_y_var":[]}
    df = pd.DataFrame(data=d)
    performance_multi_runs(args, epsilons, df, accu=accu)
    #print_epsilons_vs_acc(df, clients = chosen_clients, accu=accu)
    print_features_vs_chosen_y(df, x="epsilon", y="Gradient_network_loss")
    #print_epsilons_vs_gap(df, clients = chosen_clients, accu=accu)

def show_features_effect(args, accu="extraction_accuracy"):
    features = [100,500,1000]
    d = {"feature_number": [], "client": [], "run": [], "Decode_accu": [], "Server_accu": []}
    df = pd.DataFrame(data=d)
    performance_multi_runs_features(args, features, df, accu=accu)
    print_features_vs_acc(df, property="feature_number", accu=accu)

def show_local_steps_effect(args, accu="extraction_accuracy"):
    local_steps = [args.num_local_steps]
    d = {"local_step": [], "client": [], "run": [], "Decode_accu": [], "Server_accu": [], "Gap":[]}
    df = pd.DataFrame(data=d)
    local_model_acc = performance_multi_runs_localsteps(args, local_steps, df, accu=accu)
    print(df)
    print_features_vs_acc(df, property="local_step", accu=accu)
    #print_FL_acc_vs_iter(local_model_acc, df)

def show_batches_effect(args, accu="extraction_accuracy"):
    batches = [32,64,128,256,512]
    d = {"batch_size": [], "client": [], "run": [], "Decode_accu": [], "Server_accu": [], "Gap":[], "Gradient_network_loss":[], "Gradient_norm":[]}
    df = pd.DataFrame(data=d)
    performance_multi_runs_batches(args, batches, df, accu=accu)
    print_features_vs_chosen_y(df, x="batch_size", y="Gradient_network_loss")
    #print_features_vs_acc(df, property="batch_size", accu=accu)

if __name__ == '__main__':
    args = parse_args()
    #show_features_effect(args)
    #show_epsilons_effect(args, chosen_clients= [0,1,2,3,4],accu="extraction_accuracy")
    #show_local_steps_effect(args, accu="train_accuracy")
    show_batches_effect(args, accu="test_accuracy")
    #show_batches_effect(args, accu="extraction_accuracy")