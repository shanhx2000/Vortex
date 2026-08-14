import transformers
import torch
import torch.nn as nn
from transformers import AutoConfig
from collections import defaultdict
import os

"""
PHI_REGISTRY = {
    "l0-norm": lambda x: (x != 0).sum(dim=-1),
    "l1-norm": lambda x: x.abs().sum(dim=-1),
    "l2-norm": lambda x: torch.sqrt((x ** 2).sum(dim=-1)),
    "l_inf-norm": lambda x: x.abs().max(dim=-1).values,
}
"""

# ===== phi utilities (module internal) =====

# Registry for all supported group metrics
_PHI_REGISTRY = {
    "l0-norm": {"tag": "l0"},
    "l1-norm": {"tag": "l1"},      # default behavior
    "l2-norm": {"tag": "l2"},
    "l_inf-norm": {"tag": "linf"},
}

def _validate_phi(phi_func: str):
    """
    Validate that the provided phi function is supported.
    Args:
        phi_func (str): Name of the group metric.
    Raises:
        ValueError: If the metric is not supported.
    """
    if phi_func not in _PHI_REGISTRY:
        raise ValueError(f"Unsupported phi_func: {phi_func}")


def _phi_suffix(phi_func: str) -> str:
    """
    Return the filename suffix corresponding to the phi function.
    By design:
        - l1-norm is the default behavior and does NOT add any suffix
        - All other norms append '_phi<tag>'
    Example:
        l1-norm   -> ""
        l2-norm   -> "_phil2"
        l0-norm   -> "_phil0"
        l_inf-norm -> "_philinf"
    """
    tag = _PHI_REGISTRY[phi_func]["tag"]
    # l1 is treated as the default baseline (backward compatible)
    if tag == "l1":
        return ""
    return f"_phi{tag}"

class SparsifyFn(nn.Module):
    def __init__(self, distr, 
                 init_sparsity=None,
                 init_threshold=None, 
                 apply_prefill=True,
                 vec_length=1,
                 phi_func="l1-norm"):
        super(SparsifyFn, self).__init__()

        _validate_phi(phi_func)
        self.phi_func = phi_func

        assert init_sparsity is None or init_threshold is None, "init_sparsity and init_threshold cannot both be specified"
        if init_sparsity is not None:
            # thresh = distr.icdf(0.5 + init_sparsity/2)
            thresh = distr.abs_icdf(init_sparsity)
        elif init_threshold is not None:
            thresh = init_threshold
        else:
            init_sparsity = 0
            thresh = 0
        self.threshold = float(thresh)
        self.sparsity_level = init_sparsity if init_sparsity is not None else 0.0
        
        self.register_buffer("a", torch.tensor([thresh]).to(torch.float16))

        self.distr = distr
        self.apply_prefill = apply_prefill
        self.vec_length = vec_length

    def group_metric(self, xg):
        """
        xg: (..., D/v, v)
        return: (..., D/v, 1)
        """
        if self.phi_func == "l0-norm":
            return (xg != 0).sum(dim=-1, keepdim=True).float()
        elif self.phi_func == "l1-norm":
            return xg.abs().sum(dim=-1, keepdim=True)
        elif self.phi_func == "l2-norm":
            return torch.sqrt((xg ** 2).sum(dim=-1, keepdim=True))
        elif self.phi_func == "l_inf-norm":
            return xg.abs().max(dim=-1, keepdim=True).values
        else:
            raise ValueError(f"Unsupported phi_func: {self.phi_func}")

    def set_threshold(self, sparsity):
        # self.threshold = self.distr.icdf(0.5 + sparsity/2).item() if sparsity != 0.0 else 0.0
        self.threshold = self.distr.abs_icdf(sparsity).item() if sparsity != 0.0 else 0.0
        self.sparsity_level = sparsity

    def forward(self, x):
        # shanhx: apply to all tokens to ensure sparsity is applied. 
        # NOTE: we can + should change this to sparsify 99% of tokens instead of 50%
        # I just finished the evals for the paper at 50% before I noticed the prefill sparsification phenomenon (Section 5.4.3)
        if x.size(1) > 1 and self.apply_prefill:
            # half_seq_len = x.size(1) // 2
            half_seq_len = int(0.99 * x.size(1))
            last_context = x[:, -half_seq_len:, :]
            modified_context = self.apply(last_context)
            x = torch.cat((x[:, :-half_seq_len, :], modified_context), dim=1)
            return x
        if x.size(1) > 1 and not self.apply_prefill:
            assert False, "assert not evaluting this case."
            return x
        assert x.size(1) == 1, "supposedly x is decode only"
        return self.apply(x)

    # def apply(self, x):
    #     return x.abs().gt(self.threshold) * x
    def apply(self, x):
        """
        x: tensor of shape (..., in_dim)
        self.vec_length: group size (default=4)
        self.threshold: scalar threshold
        """
        v = self.vec_length
        *leading, D = x.shape
        assert D % v == 0, "in_dim must be divisible by vec_length"
        # reshape to (..., D/v, v)
        xg = x.view(*leading, D // v, v)
        # # compute sum of abs within each group → (..., D/v, 1)
        # group_abs_sum = xg.abs().sum(dim=-1, keepdim=True)
        # # mask groups based on threshold → boolean (..., D/v, 1)
        # mask = group_abs_sum > self.threshold
        # phi_func
        group_metric = self.group_metric(xg)
        mask = group_metric > self.threshold
        # apply mask to all v elements of each group
        x_filtered = xg * mask
        # restore original shape
        return x_filtered.view(*leading, D)

    def get_threshold(self):
        return self.threshold

    def get_sparsity(self):
        return self.sparsity_level


def interp(x, xp, fp):
    """Custom interpolation function for PyTorch tensors."""
    i = torch.searchsorted(xp, x)
    i = torch.clamp(i, 1, len(xp) - 1)    
    xp_left = xp[i - 1]
    xp_right = xp[i]
    fp_left = fp[i - 1]
    fp_right = fp[i]
    t = (x - xp_left) / (xp_right - xp_left)
    return fp_left + t * (fp_right - fp_left)


class Distribution:
    def __init__(self, file_path, hidden_type, vec_length=1, use_abs=True, phi_func="l1-norm"):
        self.file_path = file_path
        self.hidden_type = hidden_type # h1 or h2
        self.vec_length = vec_length
        self.use_abs = use_abs
        _validate_phi(phi_func)
        self.phi_func = phi_func
        # histogram = torch.load(f"{self.file_path}/histograms_v{self.vec_length}_a{self.use_abs}.pt")
        histogram = torch.load(
            f"{self.file_path}/histograms_v{self.vec_length}_a{self.use_abs}"
            f"{_phi_suffix(self.phi_func)}.pt"
        )
        self.bin_centers, self.counts = histogram[f"{self.hidden_type}_centers"], histogram[self.hidden_type]
        self.total_count = self.counts.sum()
        self.cumulative_counts = torch.cumsum(self.counts, dim=0)

    # kernel smoothing
    def pdf(self, x, bandwidth=None):
        if bandwidth is None:
            bandwidth =  1.06 * torch.std(self.bin_centers[1:-1]) * (self.total_count-2)**(-1/5)
        bin_centers = self.bin_centers.unsqueeze(1)
        if isinstance(x, float) or isinstance(x, int):
            x = torch.tensor([x])
        else:
            x = x.unsqueeze(0)
        kernel = torch.exp(-0.5 * ((x - bin_centers) / bandwidth)**2) / (bandwidth * torch.sqrt(torch.tensor(2 * torch.pi)))
        pdf = torch.sum(kernel * self.counts.unsqueeze(1), dim=0) / self.total_count
        return pdf
    
    def cdf(self, x):
        return interp(x, self.bin_centers, self.cumulative_counts / self.total_count)
    
    # # NOTE: Assumes distribution is zero mean unimodal
    # def icdf(self, q):
    #     # if q < 0.01 or q > 0.99:
    #     #     print(f"WARNING: All outliers clip to the most extreme bin")

    #     target_count = q * self.total_count
    #     idx = torch.searchsorted(self.cumulative_counts, target_count)
        
    #     if idx == 0:
    #         return self.bin_centers[0]
    #     elif idx == len(self.bin_centers):
    #         return self.bin_centers[-1]
    #     else:
    #         lower_count = self.cumulative_counts[idx - 1]
    #         upper_count = self.cumulative_counts[idx]
    #         lower_value = self.bin_centers[idx - 1]
    #         upper_value = self.bin_centers[idx]
            
    #         fraction = (target_count - lower_count) / (upper_count - lower_count)
    #         return lower_value + fraction * (upper_value - lower_value)
    
    def abs_icdf(self, q):
        assert self.use_abs, "abs_icdf called on non-abs distribution"
        target_count = q * self.total_count
        idx = torch.searchsorted(self.cumulative_counts, target_count)
        if idx == 0:
            return self.bin_centers[0]
            # return 0
        elif idx == len(self.bin_centers):
            return self.bin_centers[-1]
        else:
            lower_count = self.cumulative_counts[idx - 1]
            upper_count = self.cumulative_counts[idx]
            lower_value = self.bin_centers[idx - 1]
            upper_value = self.bin_centers[idx]
            fraction = (target_count - lower_count) / (upper_count - lower_count)
            return lower_value + fraction * (upper_value - lower_value)

class ActivationModule:
    def __init__(self, file_path, vec_length=1, use_abs=True, phi_func="l1-norm"):
        self.file_path = file_path
        self.activations = defaultdict(list)
        self.histograms = None
        self.vec_length = vec_length
        self.use_abs = use_abs
        _validate_phi(phi_func)
        self.phi_func = phi_func
        
        # store is to store stuff like position_ids in attn (for convinience, is bad code)
        self.store = {}

    def group_metric(self, xg):
        """
        xg: (seq_len, D/v, v)
        return: (seq_len, D/v)
        """
        if self.phi_func == "l0-norm":
            return (xg != 0).sum(dim=-1).float()
        elif self.phi_func == "l1-norm":
            return xg.abs().sum(dim=-1)
        elif self.phi_func == "l2-norm":
            return torch.sqrt((xg ** 2).sum(dim=-1) + 1e-12)
        elif self.phi_func == "l_inf-norm":
            return xg.abs().max(dim=-1).values
        else:
            raise ValueError(f"Unsupported phi_func: {self.phi_func}")

    def grab_activations(self, x, key):
        if not self.use_abs:
            if x.size(1) > 1:  # Check if seq_len > 1
                self.activations[key].append(x.detach().squeeze(0).cpu().float())
        elif self.use_abs:
            if x.size(1) > 1:  # seq_len > 1
                v = self.vec_length
                *leading, D = x.shape   # x may be (1, seq_len, in_dim)
                assert D % v == 0, "in_dim must be divisible by vec_length"
                # Remove batch dimension (squeeze only dim=0)
                x = x.squeeze(0)   # shape: (seq_len, in_dim)
                # Group last dim into (seq_len, D/v, v)
                xg = x.view(*x.shape[:-1], D // v, v)
                # Compute abs-sum per group → shape (seq_len, D/v)
                # abs_vec_sum = xg.abs().sum(dim=-1)
                # phi_func
                abs_vec_sum = self.group_metric(xg)
                # Store
                self.activations[key].append(abs_vec_sum.detach().cpu().float())
        else:
            raise ValueError("use_abs must be a boolean")

    def save_activations(self):
        self.activations = self.combine_activations()
        # torch.save(self.activations, f"{self.file_path}/activations_v{self.vec_length}_a{self.use_abs}.pt")
        torch.save(
            self.activations,
            f"{self.file_path}/activations_v{self.vec_length}_a{self.use_abs}"
            f"{_phi_suffix(self.phi_func)}.pt"
        )

    def load_activations(self):
        # self.activations = torch.load(f"{self.file_path}/activations_v{self.vec_length}_a{self.use_abs}.pt")
        self.activations = torch.load(
            f"{self.file_path}/activations_v{self.vec_length}_a{self.use_abs}"
            f"{_phi_suffix(self.phi_func)}.pt"
        )

    # NOTE: This doesn't store outlier activation values
    def find_histogram(self, num_bins=10000, outlier_threshold=0.01):
        if self.histograms is None:
            # for fine-grained analysis, do not combine activations
            self.activations = self.combine_activations()
            self.histograms = {}
        else:
            return self.histograms
        
        torch.cuda.empty_cache()
        for key, acts in self.activations.items():

            acts = acts.flatten().detach().to('cuda')
            acts = torch.sort(acts)[0]

            lower_bound = acts[int(outlier_threshold * len(acts))]
            upper_bound = acts[-int(outlier_threshold * len(acts))]

            acts = acts.cpu()

            main_bins = torch.linspace(lower_bound, upper_bound, num_bins - 1)
            bins = torch.cat([torch.tensor([acts[0]]), main_bins, torch.tensor([acts[-1]])])

            counts, _ = torch.histogram(acts, bins=bins)

            bin_centers = (bins[:-1] + bins[1:]) / 2

            self.histograms[key] = counts.float().cpu()
            self.histograms[f"{key}_centers"] = bin_centers.float().cpu()
        return self.histograms
    
    def save_histogram(self):
        os.makedirs(self.file_path, exist_ok=True)
        # torch.save(self.histograms, f"{self.file_path}/histograms_v{self.vec_length}_a{self.use_abs}.pt")
        torch.save(
            self.histograms,
            f"{self.file_path}/histograms_v{self.vec_length}_a{self.use_abs}"
            f"{_phi_suffix(self.phi_func)}.pt"
        )

    def combine_activations(self):
        combined_activations = {}
        for key, acts in self.activations.items():
            combined_activations[key] = torch.cat(acts, dim=0)
        return combined_activations