# import math

# import torch
# import torch.nn as nn


# class LoRALinear(nn.Module):
#     def __init__(
#         self,
#         base_layer: nn.Linear,
#         in_features,
#         out_features,
#         lora_bias=False,
#         lora_alpha=1.0,
#         lora_r=16,
#         dropout=0.0,
#         num_users=None,
#         user_emb_pos="l",
#     ):
#         super(LoRALinear, self).__init__()
#         self.base_layer = base_layer
#         self.base_layer.requires_grad_(False)
#         self.in_features = in_features
#         self.out_features = out_features
#         self.scaling = lora_alpha
#         self.r = lora_r
#         self.lora_A = nn.Linear(in_features, lora_r, bias=False)
#         self.lora_B = nn.Linear(lora_r, out_features, bias=lora_bias)
#         self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.lora_bias = lora_bias
#         self.num_users = num_users
#         if num_users is not None:
#             assert isinstance(num_users, int)
#         self.user_emb_pos = user_emb_pos
#         assert user_emb_pos in ["l", "m", "r"]
#         if self.num_users is not None:
#             if user_emb_pos == "l":
#                 self.user_embedding = nn.Embedding(num_users, in_features)
#             elif user_emb_pos == "m":
#                 self.user_embedding = nn.Embedding(num_users, lora_r)
#             else:
#                 self.user_embedding = nn.Embedding(num_users, out_features)
#         else:
#             self.user_embedding = None

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B.weight)
#         if self.lora_bias:
#             nn.init.zeros_(self.lora_B.bias)
#         if self.num_users is not None:
#             nn.init.zeros_(self.user_embedding.weight)

#     def forward(self, x: torch.Tensor, user_index: torch.Tensor = None):
#         x = x.to(self.base_layer.weight)
#         if self.num_users is not None:
#             assert user_index is not None
#             user_index = user_index.to(x)
#         result = self.base_layer(x)
#         if self.num_users is not None:
#             x = x + self.user_embedding(user_index)
#         result = result + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
#         return result


from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose
from peft.tuners.lora import LoraConfig


class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(
        self,
        base_layer: nn.Module,
        ephemeral_gpu_offload: bool = False,
        num_users=1,
        num_items=1,
        **kwargs,
    ) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.num_users = num_users
        self.num_items = num_items
        self.cur_user_index = 0
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_bias: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = (
                base_layer.num_embeddings,
                base_layer.embedding_dim,
            )
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape
                if hasattr(base_layer.weight, "ds_shape")
                else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif (
            hasattr(base_layer, "codebooks")
            and base_layer.__class__.__name__ == "QuantizedLinear"
        ):
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif (
            hasattr(base_layer, "w_bit")
            and base_layer.__class__.__name__ == "WQLinear_GEMM"
        ):
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif (
            hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear"
        ):
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(
                base_layer, "out_features"
            ):
                in_features, out_features = (
                    base_layer.in_features,
                    base_layer.out_features,
                )
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.",
                UserWarning,
            )

        self.in_features = in_features
        self.out_features = out_features

    def set_indices(self, user_indices: list[int], item_indices: list[int]) -> None:
        self.user_indices = user_indices
        self.item_indices = item_indices
        # if user_indices >= self.num_users or user_indices < 0:
        #     raise ValueError(
        #         f"`user_indices` should be in the range [0, {self.num_users}), but got {user_indices}."
        #     )
        # self.cur_user_index = user_indices
        # for active_adapter in self.active_adapters:
        #     if active_adapter in self.lora_B.keys():
        #         for i, lora_B in enumerate(self.lora_B[active_adapter]):
        #             if i != user_indices:
        #                 lora_B.weight.requires_grad = False
        #             else:
        #                 lora_B.weight.requires_grad = True

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(
                f"`r` should be a positive integer value but the value passed is {r}"
            )

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.ModuleDict(
            {
                "user": nn.ParameterDict(
                    {
                        "weight": nn.Parameter(
                            torch.zeros(
                                self.num_users,
                                self.out_features,
                                r,
                                dtype=torch.float32,
                            )
                        ),
                        "bias": (
                            nn.Parameter(
                                torch.zeros(
                                    self.num_users,
                                    self.out_features,
                                    dtype=torch.float32,
                                )
                            )
                            if lora_bias
                            else None
                        ),
                    }
                ),
                "item": nn.ParameterDict(
                    {
                        "weight": nn.Parameter(
                            torch.zeros(
                                self.num_items,
                                self.out_features,
                                r,
                                dtype=torch.float32,
                            )
                        ),
                        "bias": (
                            nn.Parameter(
                                torch.zeros(
                                    self.num_items,
                                    self.out_features,
                                    dtype=torch.float32,
                                )
                            )
                            if lora_bias
                            else None
                        ),
                    }
                ),
                "common": nn.Linear(r, self.out_features, bias=lora_bias),
            }
        )
        self.lora_B[adapter_name]["common"].weight.data.zero_()
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(
                    self.lora_A[adapter_name].weight, a=math.sqrt(5)
                )
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(
                    self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name]
                )
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            for name, param in self.lora_B[adapter_name].named_parameters():
                nn.init.zeros_(param)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])
            if self.lora_bias[adapter_name]:
                # embeddings are not supported at the moment, but still adding this for consistency
                nn.init.zeros_(self.lora_embedding_B[adapter_name].bias)

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = (
                    self.lora_alpha[active_adapter] / self.r[active_adapter]
                )
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        # DoRA is not supported (yet), check that it's not being used. Don't check "__base__", as this is the
        # placeholder for the base model.
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    # def _mixed_batch_forward(
    #     self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    # ) -> torch.Tensor:
    #     # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
    #     # extra argument that allows mixing different adapters in the same batch at inference time.
    #     result = self.base_layer(x, *args, **kwargs)
    #     torch_result_dtype = result.dtype

    #     unique_adapters = set(adapter_names)
    #     sub_batch_indices_list = []
    #     for adapter in unique_adapters:
    #         sub_batch_indices_list.append(
    #             [index for index, item in enumerate(adapter_names) if item == adapter]
    #         )

    #     for i, active_adapter in enumerate(unique_adapters):
    #         if active_adapter == "__base__":
    #             continue
    #         if active_adapter not in self.lora_A.keys():
    #             continue

    #         lora_A = self.lora_A[active_adapter]
    #         lora_B = self.lora_B[active_adapter][self.cur_user_index]
    #         dropout = self.lora_dropout[active_adapter]
    #         scaling = self.scaling[active_adapter]

    #         # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
    #         # layer output
    #         sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
    #         lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
    #         result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

    #     return result


class LoraLinear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        raise NotImplementedError(
            "暂未解决merge前后self.cur_user_index可能的不一致问题"
        )
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                orig_weights,
                                transpose(delta_weight, self.fan_in_fan_out),
                                scaling=1,
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = (
                            self.lora_magnitude_vector[active_adapter].weight
                            / weight_norm
                        )
                        dora_factor = transpose(
                            dora_factor.view(-1, 1), self.fan_in_fan_out
                        )
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights

                    if self.lora_bias[active_adapter]:
                        new_bias = (
                            base_layer.bias
                            + self.lora_B[active_adapter][self.cur_user_index].bias
                        )
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias

                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight,
                                transpose(delta_weight, self.fan_in_fan_out),
                                scaling=1,
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = (
                            self.lora_magnitude_vector[active_adapter].weight
                            / weight_norm
                        )
                        dora_factor = transpose(
                            dora_factor.view(-1, 1), self.fan_in_fan_out
                        )
                        new_weight = dora_factor * (
                            base_layer.weight.data + delta_weight
                        )
                        base_layer.weight.data = new_weight

                    if self.lora_bias[active_adapter]:
                        base_layer.bias.data += self.lora_B[active_adapter][
                            self.cur_user_index
                        ].bias

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        raise NotImplementedError(
            "暂未解决merge前后self.cur_user_index可能的不一致问题"
        )
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = (
                        self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    )
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter][
                        self.cur_user_index
                    ].bias

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        raise NotImplementedError()
        device = self.lora_B[adapter][self.cur_user_index].weight.device
        dtype = self.lora_B[adapter][self.cur_user_index].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (
            dtype == torch.float16 or dtype == torch.bfloat16
        )

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter][self.cur_user_index].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = (
            transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter][self.cur_user_index].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise NotImplementedError(
                "Mixed batch forward is not supported yet. Please use the `adapter_names` argument in the forward method."
            )
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs
            )
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                # lora_B = self.lora_B[active_adapter][self.cur_user_index]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    lora_B_input = lora_A(dropout(x))
                    assert (
                        len(self.user_indices) == len(self.item_indices)
                        and len(self.item_indices) == lora_B_input.shape[0]
                    )
                    lora_B_output = 0
                    for k in self.lora_B[active_adapter].keys():
                        if k == "user":
                            indices = self.user_indices
                        elif k == "item":
                            indices = self.item_indices
                        elif k == "common":
                            l = self.lora_B[active_adapter][k]
                            lora_B_output = lora_B_output + l(lora_B_input)
                            continue
                        indices = torch.tensor(
                            indices, dtype=torch.long, device=lora_B_input.device
                        )
                        weight = self.lora_B[active_adapter][k]["weight"][indices]
                        bias = (
                            self.lora_B[active_adapter][k]["bias"][indices]
                            if self.lora_B[active_adapter][k]["bias"] is not None
                            else 0
                        )
                        lora_B_output = lora_B_output + (
                            torch.matmul(lora_B_input, weight.transpose(1, 2)) + bias
                        )
                    result = result + lora_B_output * scaling
                else:
                    raise NotImplementedError()
                    if isinstance(dropout, nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

    @staticmethod
    def register_peft(config: LoraConfig):
        config._register_custom_module()
