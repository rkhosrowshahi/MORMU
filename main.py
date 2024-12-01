

import argparse
import os
import numpy as np
import torch
from pymoo.indicators.hv import HV

from src.dataloader import fashion_loader
from src.mia import evaluate_mia
from src.models import MLP, LeNet5
from src.learner import trainer, evaluator
from src.unlearner import evaluate_entropy_f1score, mo_unlearner
from src.utils import get_params, load_model, load_params, save_model, total_params

import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    network_type = args.net
    dataset_type = args.dataset
    output_dir = args.output_dir
    os.makedirs(f"{output_dir}/{dataset_type}/{network_type}/{args.steps}steps", exist_ok=True)

    train_loader, test_loader, forget_loader, retain_loader, shadow_train_loader, shadow_unseen_loader = None, None, None, None, None, None
    if dataset_type.lower() == 'fashion':
        num_classes = 10  # Adjust according to your number of classes
        train_loader, test_loader, forget_loader, retain_loader, shadow_train_loader, shadow_unseen_loader = fashion_loader()

    if network_type.lower() == 'mlp':
        network = MLP
        input_size = 28 * 28  # Adjust according to your data
        network_init_params = {'input_size': input_size, 'num_classes': num_classes, 'hidden_units': [64]}
    elif network_type.lower() == 'lenet':
        network = LeNet5
        network_init_params = {'num_classes': num_classes}

    orig_model = network(**network_init_params).to(device)
    dims = total_params(model=orig_model)
    print("Total number of parameters in model:", dims)
    
    orig_model_checkpoint_path = f"{output_dir}/{dataset_type}/{network_type}/{args.steps}steps/orig_model_checkpoint.pth"
    if os.path.exists(orig_model_checkpoint_path):
        print("Previous original model checkpoint found and loaded!")
        orig_model = load_model(model=orig_model, path=orig_model_checkpoint_path)
    else:
        print("Training original model...")
        trainer(model=orig_model, data_loader=train_loader, num_epochs=200)
        save_model(model=orig_model, path=orig_model_checkpoint_path)
    orig_model_test_top1 = evaluator(model=orig_model, data_loader=test_loader)
    trained_params = get_params(model=orig_model)
    print("-"*50)

    retrain_model = network(**network_init_params).to(device)
    retrain_model_checkpoint_path = f"{output_dir}/{dataset_type}/{network_type}/{args.steps}steps/retrain_model_checkpoint.pth"

    if os.path.exists(retrain_model_checkpoint_path):
        print("Previous retrained model checkpoint found and loaded!")
        retrain_model = load_model(model=retrain_model, path=retrain_model_checkpoint_path)
    else:
        print("Retraining model on retaining data...")
        trainer(model=retrain_model, data_loader=retain_loader, num_epochs=200)
        save_model(model=retrain_model, path=retrain_model_checkpoint_path)
   
    retrained_model_test_top1 = evaluator(model=retrain_model, data_loader=test_loader)
    print("-"*50)
    
    shadow_model = network(**network_init_params).to(device)
    shadow_model_checkpoint_path = f"{output_dir}/{dataset_type}/{network_type}/{args.steps}steps/shadow_model_checkpoint.pth"
    if os.path.exists(shadow_model_checkpoint_path):
        print("Previous shadow model checkpoint found and loaded!")
        shadow_model = load_model(model=shadow_model, path=shadow_model_checkpoint_path)
    else:
        print("Training shadow model on validation data...")
        trainer(model=shadow_model, data_loader=shadow_train_loader, num_epochs=200)
        save_model(model=shadow_model, path=shadow_model_checkpoint_path)
    
    print("-"*50)
    
    retrained_model_mia = evaluate_mia(target_model=retrain_model, shadow_model=shadow_model, shadow_train_loader=shadow_train_loader, shadow_unseen_loader=shadow_unseen_loader, forget_loader=forget_loader)
    
    orig_model_mia = evaluate_mia(target_model=orig_model, shadow_model=shadow_model, shadow_train_loader=shadow_train_loader, shadow_unseen_loader=shadow_unseen_loader, forget_loader=forget_loader)

    # *** CHANGE YOUR OBJECTIVE FUNCTION NAME HERE ***
    unlearner_objs = evaluate_entropy_f1score

    print("Unlearning forgetting data...")
    unleaner_algorithm = mo_unlearner(dims=dims, n_obj=2, n_steps=args.steps, 
                                    unlearner_objs=unlearner_objs, 
                                    trained_params=trained_params, model=orig_model,
                                    forget_loader=forget_loader, retain_loader=retain_loader)
    print("Unlearning finished...")
    res = unleaner_algorithm.result()
    pareto_X, pareto_F = res.X, res.F
    pareto_size = len(pareto_X)
    
    f1, f2 = np.zeros(pareto_size), np.zeros(pareto_size)
    for i in range(pareto_size):
            unlearned_model = load_params(parameters=pareto_X[i], model=orig_model, device=device)
            X_f1, X_f2 = unlearner_objs(rng=[0, 0], model=unlearned_model, forget_loader=forget_loader, retain_loader=retain_loader, device=device)
            f1[i], f2[i] = X_f1, X_f2

    pareto_F = np.column_stack([f1, f2])

    np.savez(f"{output_dir}/{dataset_type}/{network_type}/{args.steps}steps/unlearned_pareto_front_set_{args.steps}steps.npz", X=pareto_X, F=pareto_F)

    orig_model = load_params(parameters=trained_params, model=orig_model, device=device)
    orig_model_f1, orig_model_f2 = unlearner_objs(rng=[0, 0], model=orig_model, forget_loader=forget_loader, retain_loader=retain_loader, device=device)
    retrained_model_f1, retrained_model_f2 = unlearner_objs(rng=[0, 0], model=retrain_model, forget_loader=forget_loader, retain_loader=retain_loader, device=device)


    plt.figure(figsize=(5,5))
    plt.scatter(retrained_model_f1, retrained_model_f2, s=50, marker='*', label="Retrained Model")
    plt.scatter(pareto_F[:, 0], pareto_F[:, 1], s=50, label="Unlearned Pareto Optimal")
    plt.scatter(orig_model_f1, orig_model_f2, s=50, marker='^', label="Pre-trained Original Model")
    plt.xlabel(f"$f_1$, Forget Entropy")
    plt.ylabel(f"$f_2$, Retain F1 score")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_dir}/{dataset_type}/{network_type}/{args.steps}steps/unlearned_{args.steps}steps_f1_f2_plot.pdf")
    # plt.show()
    plt.close()

    ref_point = np.array([1.1, 1.1])
    ind = HV(ref_point=ref_point)
    pareto_hv_val = ind(1-pareto_F)
    retrained_hv_val = ind(1-np.array([retrained_model_f1, retrained_model_f2]))
    orig_hv_val = ind(1-np.array([orig_model_f1, orig_model_f2]))

    pareto_mia, pareto_test_top1 = np.zeros(pareto_size), np.zeros(pareto_size)
    for i in range(pareto_size):
        unlearned_model = load_params(parameters=pareto_X[i], model=orig_model, device=device)
        m = evaluate_mia(target_model=unlearned_model, shadow_model=shadow_model, shadow_train_loader=shadow_train_loader, shadow_unseen_loader=shadow_unseen_loader, forget_loader=forget_loader)
        tf1 = evaluator(unlearned_model, test_loader)
        pareto_mia[i], pareto_test_top1[i] = m, tf1
        
    pareto_F = np.column_stack([pareto_mia, pareto_test_top1])

    # retrained_model_test_top1 = evaluator(retrain_model, test_loader)
    # retrained_model_mia = evaluate_mia(target_model=retrain_model, shadow_model=shadow_model, shadow_train_loader=shadow_train_loader, shadow_unseen_loader=shadow_unseen_loader, forget_loader=forget_loader)

    # orig_model_test_top1 = evaluator(orig_model, test_loader)
    # orig_model_mia = evaluate_mia(target_model=orig_model, shadow_model=shadow_model, shadow_train_loader=shadow_train_loader, shadow_unseen_loader=shadow_unseen_loader, forget_loader=forget_loader)

    plt.figure(figsize=(5,5))
    plt.scatter(retrained_model_mia*100, retrained_model_test_top1*100, s=50, marker='*', label="Retrained Model")
    plt.scatter((pareto_mia)*100, pareto_test_top1*100, s=50, label="Unlearned Pareto Optimal")
    plt.scatter(orig_model_mia*100, orig_model_test_top1*100, s=50, marker='^', label="Pre-trained Original Model")
    plt.xlabel(f"MIA (%)")
    plt.ylabel(f"Test Top-1 (%)")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_dir}/{dataset_type}/{network_type}/{args.steps}steps/unlearned_{args.steps}steps_test_mia_plot.pdf")
    # plt.show()
    plt.close()

    results_dict = {}
    results_dict['Models'] = np.concatenate([['Original Model'], ['Retrained Model'], ['Unlearned Model']*pareto_size])
    results_dict['MIA'] = np.concatenate([[orig_model_mia], [retrained_model_mia], pareto_mia])
    results_dict['Test Top-1'] = np.concatenate([[orig_model_test_top1], [retrained_model_test_top1], pareto_test_top1])
    results_dict['HV'] = np.concatenate([[orig_hv_val], [orig_hv_val], [pareto_hv_val]*pareto_size])
    results_dict['f_1'] = np.concatenate([[orig_model_f1], [retrained_model_f1], pareto_F[:, 0]])
    results_dict['f_2'] = np.concatenate([[orig_model_f2], [retrained_model_f2], pareto_F[:, 1]])

    df = pd.DataFrame(results_dict)
    print(df)
    df.to_csv(f"{output_dir}/{dataset_type}/{network_type}/{args.steps}steps/result_table.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple command-line parser example.")

    # Positional arguments (required)
    parser.add_argument("--net", type=str, default='mlp', help="Network type. Choose between mlp or lenet")
    parser.add_argument("--dataset", type=str, default='fashion', help="Choose dataset")

    # Optional arguments
    parser.add_argument(
        "--output_dir", type=str, default="./out/mlp",help="Directory to the output file (optional)."
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of steps for MO unlearning"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output."
    )

    args = parser.parse_args()
    main(args)