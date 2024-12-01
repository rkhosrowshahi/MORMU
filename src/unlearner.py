import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score, precision_score, f1_score
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from src.utils import fitness_reshaper, initialization, load_params

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_entropy_f1score(rng, model, forget_loader, retain_loader, device):

    X_u, y_u = forget_loader.sample(rng[0])
    X_u, y_u = X_u.to(device), y_u.to(device)
    out_u = model(X_u)

    X_r, y_r = retain_loader.sample(rng[0])
    X_r, y_r = X_r.to(device), y_r.to(device)
    out_r = model(X_r)

    _, pred_u = torch.max(out_u, dim=1)
    f1 = f1_score(y_true=y_u.detach().cpu().numpy(), y_pred=pred_u.detach().cpu().numpy(), average="macro", zero_division=0.0)
    # f1 = nn.functional.cross_entropy(out_u, y_u).item()
    # Calculate the entropy of the model's predictions to encourage uncertainty
    probs = F.softmax(out_u, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
    f1 = 0 + entropy
    f1 = f1.item()

    _, pred_r = torch.max(out_r, dim=1)
    f2 = f1_score(y_true=y_r.detach().cpu().numpy(), y_pred=pred_r.detach().cpu().numpy(), average="weighted", zero_division=0.0)
    # f2 = nn.functional.cross_entropy(out_r, y_r).item()

    return f1, f2

def evaluate_3penalty_f1score(rng, model, forget_loader, retain_loader, device):

    f1, f2 = 0, 0

    X_u, y_u = forget_loader.sample(rng[0])
    X_u, y_u = X_u.to(device), y_u.to(device)
    out_u = model(X_u)

    X_r, y_r = retain_loader.sample(rng[0])
    X_r, y_r = X_r.to(device), y_r.to(device)
    out_r = model(X_r)

    return f1, f2

def mo_unlearner(dims, n_obj, n_steps, objectives, model, trained_params, forget_loader, retain_loader):
    problem = Problem(n_var=dims, n_obj=n_obj, n_constr=0, xl=np.ones(dims) * -1, xu=np.ones(dims), )

    algorithm = NSGA2(pop_size=100)

    # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
    algorithm.setup(problem, termination=('n_gen', n_steps), seed=1, verbose=False)

    # let the algorithm object never terminate and let the loop control it
    termination = NoTermination()

    # # create an algorithm object that never terminates
    # algorithm.setup(problem, termination=termination)

    import warnings
    warnings.filterwarnings('ignore')

    # fix the random seed manually
    np.random.seed(1)

    # until the algorithm has no terminated
    for n_gen in range(n_steps):
        # rng, rng_input, rng_eval = jax.random.split(rng, 3)
        rng, rng_input, rng_eval = np.random.randint(0, 10**6, size=(3, 2), dtype=int)
        # ask the algorithm for the next solution to be evaluated
        pop = algorithm.ask()

        # get the design space values of the algorithm
        pop_X = pop.get("X")
        if n_gen == 0:
            pop_X = initialization(N=len(pop_X), x0=trained_params)
            pop.set("X", pop_X)

        # implement your evluation. here ZDT1
        f1, f2 = np.zeros(len(pop_X)), np.zeros(len(pop_X))
        for i in range(len(pop_X)):
            model = load_params(parameters=pop_X[i], model=model, device=device)
            X_f1, X_f2 = objectives(rng=rng_eval, model=model, forget_loader=forget_loader, retain_loader=retain_loader, device=device)
            f1[i], f2[i] = X_f1, X_f2
        
        re_f1, re_f2 = fitness_reshaper(pop_X, f1, w_decay=0.1, norm=False, maximize=True), fitness_reshaper(pop_X, f2, w_decay=0.1, norm=False, maximize=True)
        pop_F = np.column_stack([re_f1, re_f2])

        static = StaticProblem(problem, F=pop_F)
        Evaluator().eval(static, pop)

        # returned the evaluated individuals which have been evaluated or even modified
        algorithm.tell(infills=pop)
        res = algorithm.result()

        # do same more things, printing, logging, storing or even modifying the algorithm object
        print(algorithm.n_gen, len(res.F), res.F.mean(axis=0), res.F.max(axis=0), res.F.min(axis=0))

    return algorithm