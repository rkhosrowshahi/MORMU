import numpy as np
import torch


def load_params(parameters, model, device):
    counted_params = 0
    for layer in model.parameters():
        # if not 'norm' in name:
            layer_size = layer.size()
            layer_size_numel = layer_size.numel()
            layer.data = torch.from_numpy(
                parameters[counted_params : layer_size_numel + counted_params].reshape(
                    layer_size
                )
            ).to(device=device, dtype=torch.float32)
            counted_params += layer.size().numel()
    
    return model

def get_params(model):
    return np.concatenate([p.flatten().detach().cpu().numpy() for p in model.parameters()])

def total_params(model):
    return np.sum([p.numel() for p in model.parameters()])

def compute_l2_norm(x):
    return np.nanmean(x*x, axis=1)

def compute_ranks(fitness):
    ranks = np.zeros(len(fitness))
    ranks[fitness.argsort()] = np.arange(len(fitness))
    return ranks

def centered_rank_trafo(fitness):
    y = compute_ranks(fitness)
    y /= (fitness.size) - 1
    return y

def fitness_reshaper(X, fitness, w_decay=0.1, norm=True, maximize=True):
    if maximize == True:
        fitness = 1 - fitness
    l2_fit_red = w_decay * compute_l2_norm(X)
    fitness += l2_fit_red

    if norm == True:
        fitness = centered_rank_trafo(fitness)

    return fitness

def initialization(N, x0):

    init_pop = np.random.normal(loc=x0, scale=0.01, size=(N, len(x0)))
    init_pop[0] = x0.copy()

    return init_pop

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model