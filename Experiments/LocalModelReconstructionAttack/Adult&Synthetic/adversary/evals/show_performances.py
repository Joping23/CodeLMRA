from attack.attacker import Insider_attack
import matplotlib.pyplot as plt

def performance_vs_epsilon(epsilons, clients, args):
    args_trials = args
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = ['b', 'r', 'k', 'c', 'm']
    colors_chosen = [colors[i] for i in clients]
    for client, color in zip(clients, colors_chosen):
        extract_acc_server = []
        extract_acc_decode = []
        for e in epsilons:
            if e > 10:
                args_trials.epsilon = 1.0
                args_trials.DP = False
            else:
                args_trials.DP = True
                args_trials.epsilon = e
            attacker = Insider_attack(args_trials)
            extract_acc_server.append(attacker.server_model["extraction_accuracy"][client])
            extract_acc_decode.append(attacker.decoded_models[str(client)]["extraction_accuracy"])
        print(extract_acc_server)
        ax.plot(epsilons, extract_acc_decode, '--x', color=color, label=f"decoded model for client "+str(client))
        ax.plot(epsilons, extract_acc_server, '-o', color=color, label=f"global model for client "+str(client))
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    print(labels)
    labels[-2] = 'inf'
    ax.set_xticklabels(labels)
    plt.xlabel("Epsilon")
    plt.ylabel("Extraction accuracy")
    ax.legend()
    plt.show()

def performance_vs_bz(bzs, clients, args):
    args_trials = args
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = ['b', 'r', 'k', 'c', 'm']
    colors_chosen = [colors[i] for i in clients]
    for client, color in zip(clients, colors_chosen):
        extract_acc_server = []
        extract_acc_decode = []
        for bz in bzs:
            args_trials.bz=bz
            attacker = Insider_attack(args_trials)
            extract_acc_server.append(attacker.server_model["extraction_accuracy"][client])
            extract_acc_decode.append(attacker.decoded_models[str(client)]["extraction_accuracy"])
        print(extract_acc_server)
        ax.plot(bzs, extract_acc_decode, '--x', color=color, label=f"decoded model for client "+str(client))
        ax.plot(bzs, extract_acc_server, '-o', color=color, label=f"global model for client "+str(client))
    plt.title(r"$\epsilon$="+str(args.epsilon)+" local steps="+str(args.num_local_steps))
    plt.xlabel("Batch size")
    plt.ylabel("Extraction accuracy")
    plt.xscale("log", basex=2)
    ax.legend()
    plt.show()

def performance_vs_lstp(lstps, clients, args):
    args_trials = args
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = ['b', 'r', 'k', 'c', 'm']
    colors_chosen = [colors[i] for i in clients]
    for client, color in zip(clients, colors_chosen):
        extract_acc_server = []
        extract_acc_decode = []
        for stp in lstps:
            args_trials.num_local_steps=stp
            attacker = Insider_attack(args_trials)
            extract_acc_server.append(attacker.server_model["extraction_accuracy"][client])
            extract_acc_decode.append(attacker.decoded_models[str(client)]["extraction_accuracy"])
        print(extract_acc_server)
        ax.plot(lstps, extract_acc_decode, '--x', color=color, label=f"decoded model for client "+str(client))
        ax.plot(lstps, extract_acc_server, '-o', color=color, label=f"global model for client "+str(client))
    plt.title(r"$\epsilon$="+str(args.epsilon)+" batch size="+str(args.bz))
    plt.xlabel("Local steps")
    plt.ylabel("Extraction accuracy")
    ax.legend()
    plt.show()


