from federated_learning.federated_learning_process import FederatedLearningFramework
from utils.args import parse_args
from adversary.adversary import Adversary
from attack.attacker import Insider_attack
from adversary.evals.show_performances import performance_vs_epsilon, performance_vs_bz, performance_vs_lstp

if __name__ == '__main__':
    args = parse_args()
    results_list = []
    seed = 1234
    for t in range(args.runs):
        adversary = Adversary(args)
        local_model_accuracy = []
        if not adversary.check_if_data_exists():
            FL_Process = FederatedLearningFramework(args, seed=seed+t)
            local_model_accuracy = FL_Process.launch()

        adversary.settle_for_decode_evaluation()

        adversary.check_other_benchmark()
        print("----------- Phase I: simulates the gradient networks for workers --------------------")
        #if args.adversary_ability != "personalized_attack":
        adversary.train_gradient_network()

        print("----------- Phase II: decoding the local optimum for workers --------------------")
        adversary.decode_local_models()
        if t<args.runs-1 and args.runs>1:
            results_list = adversary.save_results(results_list=results_list, local_model_accuracy=local_model_accuracy, write_tag=False)
        else:
            results_list = adversary.save_results(results_list=results_list, local_model_accuracy=local_model_accuracy, write_tag=True)
        #adversary.clean_useless_states()

    """
    #performance_vs_epsilon([1.0,5.0,10.0,12.5],[0,1,2,3,4], args)
    #performance_vs_bz([32,64,128,256,512], [3], args)
    #performance_vs_lstp([1,5,10,15,20],[3], args)
    #attacker = Insider_attack(args)
    #attacker.print_extraction_performance()
    performance_multi_runs(args)

    
    if args.experiment == "synthetic":
        attacker.attack_pair(added_test_local_data=False, even_test_data=True)
    elif args.experiment == "adult" or args.experiment == "purchase_100":
        attacker.attack_pair(added_test_local_data=False, even_test_data=False)
    else:
        raise NotImplementedError
    attacker.save_results()
    
    #attacker.attack_chosen([2,3], added_test_local_data=False, even_test_data=False)
    #attacker.attack_all(added_test_local_data=False, even_test_data=True)
    attacker.plot_attacker_performance_paper()
    #attacker.plot_embedding_optimum_model()
    #attacker.interpret_models()
    #attacker.plot_data_point_probability([2,7])
    """
