import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

from diffusers.models.attention_processor import Attention, SpatialNorm ,AttnProcessor

class PaletteLoRAAttnProcessor(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, network_alpha=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        
        self.to_q_lora = PaletteLoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = PaletteLoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = PaletteLoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = PaletteLoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None , color_palette=None
    ,**cross_attention_kwargs ):
        
        for key in cross_attention_kwargs:
            if key == "scale":
                scale = cross_attention_kwargs[key]
                del cross_attention_kwargs[key]

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states,color_palette,**cross_attention_kwargs)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states,color_palette,**cross_attention_kwargs)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states,color_palette,**cross_attention_kwargs)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states,color_palette,**cross_attention_kwargs)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
import torch.nn as nn
class PaletteLoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")
        self.in_features = in_features
        self.out_features = out_features
        self.down = nn.Linear(5*3, in_features * rank, bias=False)
        self.up = nn.Linear(5*3, rank * out_features, bias=False)
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states, **cross_attention_kwargs):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        color_palette = color_palette.reshape(1, 5*3)

        down = self.down(color_palette).reshape( self.rank,self.in_features)
        up = self.up(color_palette).reshape( self.out_features,self.rank)

        down_hidden_states = torch.matmul(hidden_states.to(dtype) , down.transpose(0,1))
        up_hidden_states = torch.matmul(down_hidden_states , up.transpose(0,1) )

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)



def show_palette(palette):
    #number of colors
    num_colors = len(palette)
    
    #show all the colors side by side on a single figure
    fig = plt.figure(figsize=(num_colors,1))
    fig , ax = plt.subplots(1,num_colors,figsize=(num_colors,1))
    fig.subplots_adjust(top=0.9,bottom=0,left=0,right=1,wspace=0.05)
    for i in range(num_colors):
        ax[i].set_axis_off()
        ax[i].imshow([[palette[i]]])    
    plt.show()

def get_dominant_colors(image_tensor, num_colors=5, max_iterations=10):
    # Get the shape of the image tensor
    height, width, _ = image_tensor.shape

    # Reshape the image tensor to a 2D array of pixels
    pixels = image_tensor.permute(2, 0, 1).reshape(3, -1).to("cuda")

    # Normalize the pixel values between 0 and 1
    #pixels_normalized = F.normalize(pixels.float(), p=2, dim=0)

    # Perform K-medians clustering
    centroids, _ = kmedians(pixels, num_colors, max_iterations)

    # Get the RGB values of the cluster centers
    
    for i in range(2):
        col_index_to_sort=i
        sorted_indices = centroids[:, col_index_to_sort].sort(stable=True)[1]
        centroids = centroids[sorted_indices]

    return centroids

def kmedians(pixels, num_clusters, max_iterations):
    # Initialize the centroids randomly
    indices = torch.randperm(pixels.size(1))[:num_clusters].to("cuda")
    centroids = pixels[:, indices]

    #centroids = compute_median(pixels, num_clusters)
    for _ in range(max_iterations):
        # Compute distances between pixels and centroids
        distances = pairwise_distances(pixels.t(), centroids.t())

        # Assign each pixel to the nearest centroid
        _, assignments = distances.min(dim=1)

        # Update centroids based on assigned pixels
        for i in range(num_clusters):
            mask = (assignments == i)
            if mask.any():
                centroids[:, i] = median(pixels[:, mask])

    return centroids.t(), assignments

def compute_median(x, num_clusters):
    # Compute the median of the colors in the image
    medians = []
    for i in range(num_clusters):
        cluster_pixels = x[:, torch.randperm(x.size(1))[:10_000]]  # Randomly sample a subset for efficiency
        median_color = median(cluster_pixels)
        medians.append(median_color)
    return torch.stack(medians, dim=1)

def pairwise_distances(x, y):
    return torch.cdist(x, y, p=1)

def median(x):
    # Sort along the first dimension
    sorted_x, _ = x.sort(dim=1)

    # Compute the median
    n = sorted_x.size(1)
    if n % 2 == 1:
        return sorted_x[:, n // 2]
    else:
        return 0.5 * (sorted_x[:, n // 2 - 1] + sorted_x[:, n // 2])


def updated_get_dominant_colors(image_tensor, num_colors=5, max_iterations=100, **kwargs):
    from kmeans_pytorch import kmeans
    flat = image_tensor.reshape(-1, image_tensor.size(-1))
    dominant_colors = kmeans(X=flat, num_clusters=num_colors,device="cuda",tol=1e-9,tqdm_flag=False)[1]
    col_index_to_sort=2
    sorted_indices = dominant_colors[:, col_index_to_sort].sort(stable=True)[1]
    dominant_colors = dominant_colors[sorted_indices]
    dominant_colors = dominant_colors.to("cpu")
    del flat
    return dominant_colors
    

class maxwell_boltzman_schedualer:
    #f(x) = base + scaling*x^2*exp(-decay*sqrt(x))
    def __init__(self,base,scaling,decay):
        self.base = base
        self.scaling = scaling
        self.decay = decay
        self.step = 0
    def __call__(self, step):
        import numpy as np
        self.step = step
        return self.base + self.scaling*(self.step**2)*np.exp(-self.decay*np.sqrt(self.step))
    
    
def show_progress_pics(images, timesteps,global_step):
    fig, axs =  plt.subplots(1,len(images),figsize=(15,5))
    for i in range(len(images)):
        axs[i].imshow(images[i][0])
        axs[i].set_title(images[i][1])
        axs[i].axis('off')
    fig.suptitle(f"global step: {global_step} , timesteps: {timesteps.cpu().item()}", fontsize=16)
    plt.show()

        
# fill attn processors
#this section is for filling the attn processors with pretrained weights
def load_palette_proccess(unet,path_to_weights):
    import os
    from collections import defaultdict
    from typing import Callable, Dict, Union

    import torch
    from diffusers.models.modeling_utils import _get_model_file
    from diffusers.utils import DIFFUSERS_CACHE, HF_HUB_OFFLINE, logging

    path = path_to_weights
    state_dict = torch.load(path, map_location="cpu")
    attn_processors = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    is_lora = all("lora" in k for k in state_dict.keys())

    if is_lora:
        lora_grouped_dict = defaultdict(dict)
        for key, value in state_dict.items():
            attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
            lora_grouped_dict[attn_processor_key][sub_key] = value

        for key, value_dict in lora_grouped_dict.items():
            rank = value_dict["to_k_lora.down.weight"].shape[0]
            rank = 4
            cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[0]
            if not (cross_attention_dim == None):
                cross_attention_dim = int(cross_attention_dim / 4)
            
            hidden_size = value_dict["to_k_lora.up.weight"].shape[0]
            
            hidden_size =int( hidden_size / 4)

            attn_processors[key] = PaletteLoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=4
            )
            attn_processors[key].load_state_dict(value_dict)

    else:
        raise ValueError(f"{model_file} does not seem to be in the correct format expected by LoRA training.")

    # set correct dtype & device
    attn_processors = {k: v.to(device=device, dtype=dtype) for k, v in attn_processors.items()}

    # set layers
    unet.set_attn_processor(attn_processors)

    return unet
