import os
import torch
from typing import List, Any

try:
    # optional import
    import matplotlib.pyplot as plt # for making figures
except:
    pass

class Color:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED =  '\033[91m'
    GRAY = '\033[90m'
    ENDC = '\033[0m'

def gelu_saturation(x:torch.Tensor):
    return x < -2 

def linear_saturation(x:torch.Tensor, threshold:float=3):
    return abs(x) > threshold 

def plot(data:list, legends:list, title:str, file_prefix:str, out_dir:str="./", xscale="linear", xlim=(-6,6), ylim=(0,1)):
    plt.figure(figsize=(20, 6)) # width and height of the plot
    for d in data:
        plt.plot(*d)
    plt.xscale(xscale)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(legends)
    plt.title(f'{title}')
    plt.savefig(os.path.join(out_dir, f"{file_prefix}.pdf"))
    plt.show()

def _get_title(title:str, prefix:str, iter_num:int, loss:float=None):
    if iter_num != None:
        title += f": iter = {iter_num}"
        prefix = f"iter{iter_num:05d}-{prefix}"
    if loss != None:
        title += f", train loss = {loss:.4f}"
    return (title, prefix)

def weights(model:Any, activation_type:str="linear", title:str="Pre-activation Distribution", iter_num:int=None, out_dir:str=".", loss:float=None, noplot:bool=False):
    legends = []
    data = []
    layers = [('linear', model.module_dict['linear'].weight)]
    for i, (name, layer) in enumerate(layers):
        stats = f'mean = {layer.mean():+.2e}, std = {layer.std():.2e}, min = {layer.min():.3f}, max = {layer.max():.3f}, saturation = {linear_saturation(layer).float().mean()*100:.2f}%'
        print(Color.RED + f'Weights: {name} {i} ({activation_type}): {stats}' + Color.ENDC)
        hy, hx = torch.histogram(layer, density=True)
        data.append((hx[:-1].detach(), hy.detach()))
        legends.append(f'{name} {i} ({activation_type})')
    title, prefix = _get_title(title, 'Parameters', iter_num, loss)
    if not noplot:
        plot(data, legends, f"{title}, {stats}", prefix, out_dir)

def similarity(model:Any, title:str="Similarity Distribution", iter_num:int=None, out_dir:str=".", loss:float=None, noplot:bool=False):
    raw = model.logits
    mean = torch.mean(raw, dim=(0,1))
    stats = f'mean = {mean.mean():+.2f}, std = {mean.std():.2f}'
    print(f'Logits: {stats}')
    hy, hx = torch.histogram(raw, density=True)
    data = [(hx[:-1].detach(), hy.detach())]
    legend = ['Logits']
    title, prefix = _get_title(title, "logits", iter_num, loss)
    if not noplot:
        plot(data, legend, f"{title}, {stats}", prefix, out_dir, xlim=(-25, 25))

def grad(model:Any, title:str="Gradient Distribution", iter_num:int=None, out_dir:str=".", loss:float=None, noplot:bool=False):
    layers = [('linear.weight', model.module_dict['linear'].weight.grad)]
    legends = []
    data = []
    for name, layer in layers:
        if layer != None:
            stats = f'mean = {layer.mean().item():+.4e}, std = {layer.std().item():.4e}'
            print(Color.RED + f'Gradients: "{name}": {stats}' + Color.ENDC)
            hy, hx = torch.histogram(layer, density=True)
            data.append((hx[:-1].detach(), hy.detach()))
            legends.append(f'Layer "{name}"')

    title, prefix = _get_title(title, "gradients", iter_num, loss)
    if not noplot:
        plot(data, legends, f"{title}, {stats}", prefix, out_dir, xscale="log", xlim=(1e-5, 1e-1), ylim=(0, 1000))
