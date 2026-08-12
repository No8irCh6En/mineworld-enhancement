import torch
from typing import Optional, Dict, Tuple, List, Any
from torch.nn.attention import SDPBackend
import inspect, sys, traceback
import os
import time


def sample_top_k(logits, temperature: float = 1.0, top_k: Optional[int] = None, vocab_size=8192):
    """
    Sample from the logits using top-k sampling.
    Source: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    # logits: [batch_size, seq_len, vocab_size]
    if temperature == 0.0:
        idx_next = torch.argmax(logits[:, -1, :vocab_size], dim=-1, keepdim=True)
    else:
        probs = logits_to_probs(logits[:, -1, :vocab_size], temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next

def multinomial_sample_one_no_sync(probs_sort, dtype=torch.int):
    """
    Multinomial sampling without a cuda synchronization.
    Source: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=dtype)

def logits_to_probs(
    logits,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample_top_p(logits, temperature, top_p, vocab_size=8192):
    probs = torch.softmax(logits[:, -1, :vocab_size] / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial_sample_one_no_sync(probs_sort, dtype=torch.int64)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def sample_n_top_p(logits, temperature, top_p, vocab_size=8192):
    probs = torch.softmax(logits[:, :, :vocab_size] / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial_sample_one_no_sync(probs_sort, dtype=torch.int64)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token.clone()


def sample_n_top_k(logits, temperature: float = 1.0, top_k: Optional[int] = None, vocab_size=8192):
    if temperature == 0.0:
        # Modify for multiple logits (n items)
        idx_next = torch.argmax(logits[:, :, :vocab_size], dim=-1, keepdim=True)  # Use all n logits for top-k
        probs = None
    else:
        probs = logits_to_n_probs(logits[:, :, :vocab_size], temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)

    return idx_next

def logits_to_n_probs(
    logits,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def get_logits_from_output(model_out):
    """
    从 model(...) 的返回值中抽取 logits tensor。
    支持：直接 tensor、(logits, aux)、dict with "logits"。
    如无法抽取则抛出有意义的异常并打印部分上下文。
    """
    # 直接是 tensor
    if isinstance(model_out, torch.Tensor):
        return model_out
    # dict 返回
    if isinstance(model_out, dict):
        if "logits" in model_out:
            return model_out["logits"]
        # fallback: first tensor-like value
        for v in model_out.values():
            if isinstance(v, torch.Tensor):
                return v
        # 无法找到 tensor，打印上下文再报错
        # print("[DEBUG] model returned dict but no tensor-like value found. keys:", list(model_out.keys()))
        raise ValueError("model(...) returned dict but no logits tensor found")
    # tuple/list 返回，取第一个元素
    if isinstance(model_out, (tuple, list)):
        first = model_out[0]
        if isinstance(first, torch.Tensor):
            return first
        # 如果第一个不是 tensor，尝试查找第一个 tensor
        for elem in model_out:
            if isinstance(elem, torch.Tensor):
                return elem
        print("[DEBUG] model returned tuple/list but no tensor-like element found. repr:", repr(model_out)[:1000])
        raise ValueError("model(...) returned tuple/list but no logits tensor found")
    # 其它类型
    print("[DEBUG] model returned unexpected type:", type(model_out))
    raise ValueError("Cannot extract logits from model(...) return type: " + str(type(model_out)))

def decode_one_token(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    """
    Decode a single token from the autoregressive model.
    """
    logits = model(input_ids=input_ids, position_ids=position_ids)
    # logits = get_logits_from_output(logits)
    if top_p is not None:
        return sample_top_p(logits, temperature=temperature, top_p=top_p)
    else:
        return sample_top_k(logits, temperature=temperature, top_k=top_k)
    
def decode_some_token(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    """
    Decode multi token from the autoregressive model.
    """
    logits = model(input_ids=input_ids, position_ids=position_ids)
    # logits = get_logits_from_output(logits)
    if top_p is not None:
        return sample_n_top_p(logits, temperature=temperature, top_p=top_p)
    else:
        return sample_n_top_k(logits, temperature=temperature, top_k=top_k)
    
def decode_n_tokens(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    num_generate_tokens: int,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = None,
    decode_one_token_function=decode_one_token,
    pixnum: int = 336,
    actnum: int = 11,
    **kwargs,
):
    """
    Decode n tokens from the autoregressive model.
    Adapted from https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    new_tokens = [input_ids]
    pos_ = position_ids
    # print(f"DEBUG pos_:{pos_}")
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"

    for t in range(num_generate_tokens):
        with torch.nn.attention.sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            # start_time = time.perf_counter()
            next_token = decode_one_token_function(
                model,
                input_ids=input_ids,
                position_ids=position_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            # print(f"[DEBUG] t:{t}, input_ids shape: {input_ids.shape}, decode time: {time.perf_counter() - start_time:.3f}s")
            pos_ += 1
            position_ids = pos_
            new_tokens.append(next_token.clone())
            input_ids = next_token.clone()

            if (pos_ - pixnum + 1) % (actnum + pixnum) == 0 and t+2 < num_generate_tokens:
                action = kwargs["action"][ (t+2) // pixnum ]
                input_ids = torch.cat((input_ids, action), dim=-1)
                position_ids = torch.tensor([pos_ + _ for _ in range(actnum+1)], dtype=torch.long, device="cuda")
                pos_ += actnum

    return new_tokens

def decode_n_tokens_for_gradio(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    num_generate_tokens: int,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = None,
    decode_one_token_function=decode_one_token,
):
    """
    Decode n tokens from the autoregressive model.
    Adapted from https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    new_tokens = []
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"
    position_id = position_ids[-1].unsqueeze(0)
    assert num_generate_tokens % 336 == 1, "should be pixnum x n + 1 to fill kvcache"
    for t in range(num_generate_tokens):
        with torch.nn.attention.sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token_function(
                model,
                input_ids=input_ids,
                position_ids=position_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            position_id += 1
            position_ids = position_id
            new_tokens.append(next_token.clone())
            input_ids = next_token.clone()
    return new_tokens[:-1], position_id

def prefill(
    model,
    input_ids: torch.Tensor = None,
    position_ids: torch.Tensor = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.8,
    **kwargs,
):
    logits = model(input_ids=input_ids, position_ids=position_ids)
    # logits = get_logits_from_output(logits)
    # Only top-p or top-k can be provided
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"
    if top_p is not None:
        return sample_top_p(logits, temperature=temperature, top_p=top_p)
    else:
        return sample_top_k(logits, temperature=temperature, top_k=top_k)

def img_diagd_prepare_inputs(
    ongoing_row_list,
    row_token_num,
    ongoing_input,
    prompt,
    imagenum,
    pixnum: int = 336,
    actnum: int = 11,
    columnnum: int = 24,
    promptlen: int = 347,
    **kwargs
):
    position_ids = []
    
    for i in ongoing_row_list:
        global_idx = promptlen + i * columnnum + row_token_num[i] - 1 + (imagenum - 1) * (pixnum + actnum)
        position_ids.append(global_idx)

    if row_token_num[ongoing_row_list[-1]] == 0:
        append_policy = kwargs.get("append_policy", True)
        if append_policy:
            idx_in_input_ids = ongoing_row_list[-1] * columnnum - 1
            ongoing_input.append(prompt[:, idx_in_input_ids].unsqueeze(-1))
        else:
            ongoing_input.append(ongoing_input[-1])

    input_ids = torch.cat(ongoing_input, dim=1)
    position_ids = torch.tensor(position_ids, device="cuda")

    return input_ids, position_ids

def img_diagd_decode_n_tokens(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    num_generate_tokens: int,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = None,
    decode_some_token_function=decode_some_token,
    pixnum: int = 336,
    actnum: int = 11,
    columnnum: int = 24,
    rownum: int = 14,
    windowsize: int = 2,
    promptlen: int = 347,
    **kwargs,
):
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"

    imagenum = 1
    cur_len = 1
    num_generate_tokens += 1
    prompt = kwargs.pop("prompt", None) 
    new_tokens = [input_ids.clone()]
    row_token_num = torch.zeros((rownum,), dtype=torch.long, device="cuda")
    row_token_num[0] += 1 
    ongoing_row_list = [0]
    ongoing_input = [input_ids.clone()]
    # print(f"[DEBUG] ongoing_input: {ongoing_input}")

    while True:
        if cur_len >= num_generate_tokens:
            break

        if cur_len % pixnum == 0 :#and image_start_token_id_index is None: 
            imagenum += 1
            action = kwargs["action"][cur_len // pixnum]
            ongoing_input.append(action)
            input_id = torch.cat(ongoing_input, dim=-1)
            position_ids = torch.arange(imagenum * (pixnum+actnum) - actnum - 1, imagenum * (pixnum+actnum), device="cuda")
            # print(f"[DEBUG] input_id: {input_id}, position_ids: {position_ids}")

        image_token_num = cur_len % pixnum

        if image_token_num == 1 and row_token_num[0] == windowsize:
            ongoing_row_list.append(1)
        start_time = time.perf_counter()
        if image_token_num >= 1:
            input_id, position_ids = img_diagd_prepare_inputs(ongoing_row_list=ongoing_row_list, ongoing_input = ongoing_input, imagenum=imagenum, row_token_num=row_token_num, promptlen=promptlen, prompt=prompt,**kwargs)  
            # print(f"[DEBUG] after img_diagd_prepare input_id: {input_id}, position_ids: {position_ids}")
            
        num_new_tokens = input_id.shape[1] if len(ongoing_row_list) > 0 else 1

        with torch.nn.attention.sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_some_token_function(
                model,
                input_ids=input_id,
                position_ids=position_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            ongoing_input = []
            
        if len(ongoing_row_list) == 0:
            cur_len += 1
            ongoing_input.append(next_token[:,-1].clone())
            new_tokens.append(next_token[:,-1].clone())
            ongoing_row_list.append(0)
            row_token_num[0] += 1 
        else:
            need_remove_row = None
            cur_len += num_new_tokens
            for i in range(num_new_tokens):
                position_in_new_tokens = torch.sum(row_token_num[:(ongoing_row_list[i] + 1)], dim=0) + (imagenum - 1) * pixnum 
                new_tokens.insert(position_in_new_tokens, next_token[:,i].clone())
                ongoing_input.append(next_token[:,i].clone())
                row_token_num[ongoing_row_list[i]] += 1

                if row_token_num[ongoing_row_list[i]] == windowsize and ongoing_row_list[i] < rownum - 1:
                    ongoing_row_list.append(ongoing_row_list[i]+1)

                elif ongoing_row_list[i] == rownum - 1 and row_token_num[ongoing_row_list[i]] == columnnum:
                    row_token_num = torch.zeros((rownum,), dtype=torch.long, device="cuda")
                    ongoing_row_list = []
                    ongoing_input = [next_token[:,i]]
                    need_remove_row = None
                    break

                if row_token_num[ongoing_row_list[i]] == columnnum: ## this row is done
                    ongoing_input.pop()
                    need_remove_row = ongoing_row_list[i]

            if need_remove_row is not None:
                ongoing_row_list.remove(need_remove_row)
    return new_tokens

def img_diagd_prepare_inputs_for_gradio(
    ongoing_row_list,
    row_token_num,
    ongoing_input,
    pixnum: int = 336,
    actnum: int = 11,
    columnnum: int = 24,
    promptlen: int = 347,
):
    position_ids = []
    
    for i in ongoing_row_list:
        global_idx = promptlen + i * columnnum + row_token_num[i] - 1
        position_ids.append(global_idx)

    if row_token_num[ongoing_row_list[-1]] == 0:
        ongoing_input.append(ongoing_input[-1])

    input_ids = torch.cat(ongoing_input, dim=1)
    position_ids = torch.tensor(position_ids, device="cuda")

    return input_ids, position_ids

def img_diagd_decode_n_token_for_gradio(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    num_generate_tokens: int,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = None,
    decode_some_token_function=decode_some_token,
    pixnum: int = 336,
    columnnum: int = 24,
    rownum: int = 14,
    windowsize: int = 2,
):
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"

    cur_len = 0
    promptlen = position_ids[-1] + 1
    
    new_tokens = []
    row_token_num = torch.zeros((rownum,), dtype=torch.long, device="cuda")
    ongoing_row_list = []
    ongoing_input = []
    
    while True:
        if cur_len == num_generate_tokens:
            break

        image_token_num = cur_len

        if image_token_num == 1 and row_token_num[0] == windowsize:
            ongoing_row_list.append(1)
        if image_token_num == 0:
            input_id = input_ids

        if image_token_num >=1:
            input_id, position_ids = img_diagd_prepare_inputs_for_gradio(ongoing_row_list=ongoing_row_list, ongoing_input = ongoing_input, row_token_num=row_token_num, promptlen=promptlen)  
            
        num_new_tokens = input_id.shape[1] if len(ongoing_row_list) > 0 else 1
        with torch.nn.attention.sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_some_token_function(
                model,
                input_ids=input_id,
                position_ids=position_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            ongoing_input = []
        if len(ongoing_row_list) == 0:
            cur_len += 1
            ongoing_input.append(next_token[:,-1].clone())
            new_tokens.append(next_token[:,-1].clone())
            ongoing_row_list.append(0)
            row_token_num[0] += 1 
        else:
            need_remove_row = None
            cur_len += num_new_tokens
            for i in range(num_new_tokens):
                position_in_new_tokens = torch.sum(row_token_num[:(ongoing_row_list[i] + 1)], dim=0)
                new_tokens.insert(position_in_new_tokens, next_token[:,i].clone())
                ongoing_input.append(next_token[:,i].clone())
                row_token_num[ongoing_row_list[i]] += 1

                if row_token_num[ongoing_row_list[i]] == windowsize and ongoing_row_list[i] < rownum - 1:
                    ongoing_row_list.append(ongoing_row_list[i]+1)

                elif ongoing_row_list[i] == rownum - 1 and row_token_num[ongoing_row_list[i]] == columnnum:
                    row_token_num = torch.zeros((rownum,), dtype=torch.long, device="cuda")
                    ongoing_row_list = []
                    ongoing_input = [next_token[:,i]]
                    need_remove_row = None
                    break

                if row_token_num[ongoing_row_list[i]] == columnnum: ## this row is done
                    ongoing_input.pop()
                    need_remove_row = ongoing_row_list[i]

            if need_remove_row is not None:
                ongoing_row_list.remove(need_remove_row)

    with torch.nn.attention.sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            _ = decode_some_token_function(
                model,
                input_ids=next_token[:,-1],
                position_ids=position_ids+1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
    
    return new_tokens, position_ids+2



def vid_diagd_prepare_inputs(
    ongoing_row_list_v,
    row_token_num_v,
    ongoing_input_v,
    prompt,
    pixnum: int = 336,
    actnum: int = 11,
    rownum: int = 14,
    columnnum: int = 24,
    promptlen: int = 347,
    **kwargs
):
    new_frame = False
    position_ids = []

    for i in ongoing_row_list_v:
        global_idx = promptlen + i * columnnum + row_token_num_v[i // rownum][i % rownum] -1 + (i // rownum) * actnum
        position_ids.append(global_idx)

    lastrow = ongoing_row_list_v[-1]
    if lastrow % rownum == 0 and row_token_num_v[lastrow // rownum][lastrow % rownum] == 0:
        # WARNING
        action = kwargs["action"][lastrow // rownum]
        ongoing_input_v.append(action)
        position_ids.pop()
        pos_act = torch.arange( promptlen + (lastrow // rownum) * (pixnum+actnum) - actnum, promptlen + (lastrow // rownum) * (pixnum+actnum), device="cuda")
        position_ids.extend(pos_act.unbind())
        new_frame = True
    elif row_token_num_v[lastrow // rownum][lastrow % rownum] == 0:
        append_policy = kwargs.get("append_policy", True)
        if append_policy:
            idx_in_input_ids = (lastrow % rownum) * columnnum - 1
            ongoing_input_v.append(prompt[:, idx_in_input_ids].unsqueeze(-1))
        else:
            ongoing_input_v.append(ongoing_input_v[-1])

    input_ids = torch.cat(ongoing_input_v, dim=1)
    position_ids = torch.tensor(position_ids, device="cuda")

    return input_ids, position_ids, new_frame

def video_diagd_decode_n_tokens(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    num_generate_tokens: int,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = None,
    decode_some_token_function=decode_some_token,
    pixnum: int = 336,
    actnum: int = 11,
    columnnum: int = 24,
    rownum: int = 14,
    windowsize: int = 2,
    promptlen: int = 347,
    **kwargs,
):
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"

    cur_len = 1
    num_generate_tokens += 1
    prompt = kwargs.pop("prompt", None) 
    new_tokens = [input_ids.clone()]
    row_token_num_v = []
    ongoing_row_list_v = [0]
    row_token_num_v.append(torch.zeros((rownum,), dtype=torch.long, device="cuda"))
    row_token_num_v[0][0] += 1
    if row_token_num_v[0][0] == windowsize:
        ongoing_row_list_v.append(1)

    ongoing_input_v = [input_ids.clone()]

    while True:
        if cur_len >= num_generate_tokens:
            break


        input_id, position_ids, new_frame = vid_diagd_prepare_inputs(ongoing_row_list_v=ongoing_row_list_v, ongoing_input_v = ongoing_input_v, row_token_num_v=row_token_num_v, promptlen=promptlen, prompt=prompt, **kwargs)  
            
        num_new_tokens = input_id.shape[1]

        with torch.nn.attention.sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_some_token_function(
                model,
                input_ids=input_id,
                position_ids=position_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            ongoing_input_v = []
            if new_frame:
                next_token = torch.cat([next_token[:,:-actnum], next_token[:,-1:]], dim=1)
                num_new_tokens = num_new_tokens - actnum + 1
            
        need_remove_row = None

        cur_len += num_new_tokens
        for i in range(num_new_tokens):
            last_frame = torch.stack(row_token_num_v[:ongoing_row_list_v[i] // rownum]).sum() if ongoing_row_list_v[i] // rownum > 0 else torch.tensor(0, dtype=torch.long, device="cuda")
            position_in_new_tokens = last_frame + torch.sum(row_token_num_v[ongoing_row_list_v[i] // rownum][:(ongoing_row_list_v[i] % rownum + 1)], dim=0)
                
            new_tokens.insert(position_in_new_tokens, next_token[:,i].clone())
            ongoing_input_v.append(next_token[:,i].clone())
            row_token_num_v[ongoing_row_list_v[i] // rownum][ongoing_row_list_v[i] % rownum] += 1

            # WARNING
            if row_token_num_v[ongoing_row_list_v[i] // rownum][ongoing_row_list_v[i] % rownum] == windowsize and ongoing_row_list_v[i] < rownum * (num_generate_tokens//pixnum) - 1:
                ongoing_row_list_v.append(ongoing_row_list_v[i]+1)
                if ongoing_row_list_v[-1] % rownum == 0:
                    row_token_num_v.append(torch.zeros((rownum,), dtype=torch.long, device="cuda"))
            if row_token_num_v[ongoing_row_list_v[i] // rownum][ongoing_row_list_v[i] % rownum] == columnnum:
                ongoing_input_v.pop()
                need_remove_row = ongoing_row_list_v[i]

        if need_remove_row is not None:
            ongoing_row_list_v.remove(need_remove_row)
    return new_tokens

    


def speculative_decoding_step(
    model, 
    input_ids, 
    position_ids, 
    temperature, 
    top_k, 
    top_p, 
    step
):
    """
    执行一次 Speculative Forward 和 Sampling。
    这个函数是纯 Tensor 操作，适合 torch.compile。
    """
    # 1. Forward
    logits = model(input_ids=input_ids, position_ids=position_ids)

    # 2. Slicing (切片)
    relevant_logits = logits[:, -(step + 1):,:]
    
    # 3. Sampling (采样)
    if top_p is not None:
        candidates = sample_n_top_p(relevant_logits, temperature, top_p)
    else:
        candidates = sample_n_top_k(relevant_logits, temperature, top_k)
        
    return candidates.view(-1)

# -------------------------------------------------#

def speculative_img_diagd_prepare_inputs(
    main_state,     # (row_list, row_tokens, last_tokens_list)
    spec_state,     # (row_list, row_tokens, last_tokens_list)
    prompt,
    imagenum_main,
    spec_batch_size: int,
    pixnum: int = 336,
    actnum: int = 11,
    columnnum: int = 24,
    promptlen: int = 347,
    **kwargs
):
    """
    Simplified Mixed-State Input Preparation for Staggered Decoding.
    Removes redundancy and draft logic.
    """
    # Debug prints in this function run once per decode step and trigger
    # GPU->CPU syncs, which dominate the runtime. Gate them behind a flag.
    _DEBUG = os.environ.get("DIAGD_DEBUG", "0") == "1"
    start_time = time.perf_counter()
    row_list_0, row_tok_0, last_tok_0 = main_state
    row_list_1, row_tok_1, last_tok_1 = spec_state
    append_policy = kwargs.get("append_policy", True)
    end_time = time.perf_counter()
    if _DEBUG:
        print(f"[DEBUG] Unpacking states: {end_time - start_time:.6f} seconds")

    # ---- lightweight cache ----
    cache = getattr(speculative_img_diagd_prepare_inputs, "_cache", None)
    if cache is None or cache.get("columnnum") != columnnum:
        cache = {
            "columnnum": columnnum,
            "prompt_idx": [r * columnnum - 1 for r in range(1024)],
            "row_base": [r * columnnum - 1 for r in range(1024)],
        }
        speculative_img_diagd_prepare_inputs._cache = cache

    def _ensure_cache_size(n):
        if n > len(cache["prompt_idx"]):
            start = len(cache["prompt_idx"])
            cache["prompt_idx"].extend([r * columnnum - 1 for r in range(start, n)])
            cache["row_base"].extend([r * columnnum - 1 for r in range(start, n)])

    def _prepare_stream_batch(row_list, row_tok, last_tok, frame_offset, b_size):
        """Helper to prepare inputs for a single stream (Main or Spec)."""

        if not row_list:
            return torch.empty(b_size, 0, dtype=torch.long, device="cuda"), \
                   torch.empty(0, dtype=torch.long, device="cuda")

        _ensure_cache_size(max(row_list) + 1)

        pos_ids, inputs = [], []
        for r in row_list:
            global_idx = cache["row_base"][r] + row_tok[r] + frame_offset
            pos_ids.append(global_idx)

            if last_tok[r] is not None:
                token = last_tok[r]
                inputs.append(token)
            else:
                token = None
                if append_policy and prompt is not None:
                    idx = cache["prompt_idx"][r]
                    if 0 <= idx < prompt.size(1):
                        token = prompt[:, idx].unsqueeze(-1)
                        if b_size > 1: token = token.expand(b_size, -1)

                if token is None:
                    default = torch.zeros(b_size, 1, device="cuda", dtype=torch.long)
                    prev = last_tok[r - 1] if r - 1 >= 0 else None
                    token = prev if prev is not None else default
                inputs.append(token)

        input_tensor = torch.cat(inputs, dim=1)
        pos_tensor = torch.tensor(pos_ids, dtype=torch.long, device="cuda")

        return input_tensor, pos_tensor

    # === 1. Prepare Main Stream (Batch 0, Frame T) ===
    if _DEBUG:
        print("---------Main Stream Preparation---------")
        print(f"[DEBUG] row_list_0: {row_list_0}, last_tok_0 len: {len(last_tok_0)}")

    input_0, pos_0 = _prepare_stream_batch(
        row_list_0, row_tok_0, last_tok_0,
        frame_offset=promptlen + (imagenum_main - 1) * (pixnum + actnum),
        b_size=1
    )

    if _DEBUG:
        print("---------Main Stream Preparation End---------")

    # === 2. Prepare Spec Stream (Batch 1..N, Frame T+1) ===
    input_1, pos_1 = _prepare_stream_batch(
        row_list_1, row_tok_1, last_tok_1,
        frame_offset=promptlen + imagenum_main * (pixnum + actnum),
        b_size=spec_batch_size
    )

    return input_0, pos_0, input_1, pos_1


def speculative_img_diagd_decode_n_tokens(
    model,
    input_ids: torch.Tensor, 
    position_ids: torch.Tensor,
    num_generate_tokens: int,
    draft_func = None,
    action_pred_func = None,
    prefill_func = prefill,
    stagger_steps: int = 5,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = None,
    decode_some_token_function=decode_some_token,
    pixnum: int = 336,
    actnum: int = 11,
    columnnum: int = 24,
    rownum: int = 14,
    windowsize: int = 2,
    promptlen: int = 347,
    num_candidates: int = 5,
    spec_len: int = 2,
    action_prev: int = 4,
    **kwargs,
):
    """
    Staggered Speculative DiagD with Draft Prefill.
    """
    assert (top_p is None or top_k is None), "Only one of top-p or top-k can be provided"
    
    # 1. Expand KV Cache 
    model.prepare_parallel_speculation(num_candidates)
    action_seq = kwargs.get("action", None)

    # --- Prefill the prompt to populate KV cache ---
    # This matches the non-speculative flow: img_diagd_generate calls prefill
    # to process the prompt (image_input + first_action) and get the first pixel token.
    prompt = kwargs.pop("prompt", None)
    PERIOD = pixnum + actnum
    
    if prompt is not None:
        # prompt already includes the concatenated first action
        actual_prompt_len = prompt.shape[1]
        prompt_pos = torch.arange(0, actual_prompt_len, device="cuda")
        # Prefill populates KV cache at positions 0..prompt_len-1 and returns first pixel token
        first_pixel_token = prefill_func(
            model, input_ids=prompt, position_ids=prompt_pos,
            temperature=temperature, top_k=top_k, top_p=top_p
        )  # shape: [1, 1]
        # current_global_pos points to the NEXT position after first pixel token
        current_global_pos = actual_prompt_len + 1
    else:
        current_global_pos = position_ids[-1].item() + 1
        first_pixel_token = None
    
    # Sync Main's KV cache to all Spec slots for the prefill'd positions
    model.expand_kv_cache(0, current_global_pos)

    diag_schedules = [
        _get_diag_schedule(rownum, columnnum, windowsize, frame_idx=i)
        for i in range(max(1, spec_len))
    ]
    ptr_heads = [0 for _ in range(len(diag_schedules))]
    # When prefill already generated the first pixel token, skip schedule step 0
    # (which expects row_tok[0]=0). Start from step 1 where row_tok[0]=1 is expected.
    if first_pixel_token is not None:
        ptr_heads[0] = 1
    
    # 2. State Initialization (Main Stream)
    state_row_lists = [[0] for _ in range(spec_len)]
    state_row_tokens_lists = [torch.zeros((rownum,), dtype=torch.long, device="cuda") for _ in range(spec_len)]
    state_row_last_tokens = [[None for _ in range(rownum)] for _ in range(spec_len)]

    # Initialize with the first pixel token from prefill
    if first_pixel_token is not None:
        state_row_last_tokens[0][0] = first_pixel_token
        state_row_tokens_lists[0][0] = 1  # row 0 has 1 token
        if state_row_tokens_lists[0][0] == windowsize:
            state_row_lists[0].append(1)
    else:
        if state_row_tokens_lists[0][0] == windowsize:
            state_row_lists[0].append(1)

    imagenum_main = 1
    num_generate_tokens += 1
    state_0_len = 1 if first_pixel_token is not None else 0
    state_1_len = 0  # Track Spec Stream Progress
    current_step = 0 
    spec_active = False
    first_iter = True

    all_tokens_main = [input_ids.squeeze(0).tolist()]
    if first_pixel_token is not None:
        # Store first pixel token for draft_func reference
        all_tokens_main.append(first_pixel_token.squeeze(0).tolist())
    new_tokens_spec = [[] for _ in range(num_candidates)]
    new_tokens_main = []
    state_1_last_tokens_setup = None
    if action_seq is not None and len(action_seq) > 0:
        last_token = action_seq[0][:, -1:] if action_seq[0].dim() == 2 else action_seq[0][-1:].view(1, 1)
        state_1_last_tokens_setup = last_token.clone().expand(num_candidates, -1).contiguous()

    prev_action_candidates = None 
    iter_start_time = time.perf_counter()
    frame_completed = False  # set True when Main finishes a frame (row list becomes empty)
    loop_counter = 0

    while True:
        loop_counter += 1
        if state_0_len >= num_generate_tokens:
            break
        
        if os.environ.get("DIAGD_DEBUG", "0") == "1" and loop_counter % 10 == 0:
            print(f"[LOOP] iter={loop_counter} state_0_len={state_0_len} state_1_len={state_1_len} row0={state_row_lists[0]} row1={state_row_lists[1]} frame_completed={frame_completed} spec_active={spec_active} draft_done={(state_1_len % pixnum == 0)}")
        
        # --- Frame Boundary & Sync Logic ---
        # Main frame is done when its schedule has been consumed (row list empty).
        # restart now happens AFTER verification, so the empty row list persists
        # until the boundary is handled.
        main_frame_done = (state_0_len > 0 and len(state_row_lists[0]) == 0)
        # Spec frame is done when its schedule has been consumed (row list empty).
        draft_frame_done = (len(state_row_lists[1]) == 0)
        
        # 1. Wait for Spec: If Main is done but Spec (if active) is not, we pause Main
        if main_frame_done and (not draft_frame_done) and (not first_iter):
            main_active_now = False
        else:
            main_active_now = True
        first_iter = False
        
        if main_frame_done and main_active_now:
            iter_end_time = time.perf_counter()
            if os.environ.get("DIAGD_DEBUG", "0") == "1":
                print(f"[HINT] one frame( or two) time: {iter_end_time - iter_start_time:.6f} seconds")
            iter_start_time = time.perf_counter()
            # --- Verification ---

            # gt_action_idx = imagenum_main (action for the NEXT frame)
            # e.g. after frame 1 is done, we need action_1 (index 1) to transition to frame 2
            gt_action_idx = imagenum_main


            if gt_action_idx < len(kwargs["action"]):
                gt_action = kwargs["action"][gt_action_idx].to("cuda").view(1, actnum)
                # print(f"[DEBUG] GT Action for Frame {gt_action_idx}: {gt_action}")
            else:
                gt_action = torch.zeros((1, actnum), device="cuda", dtype=torch.long)
            
            last_tok_0 = [None for _ in range(rownum)]
            last_tok_0[0] = gt_action[:, -1:]
            state_row_last_tokens[0] = last_tok_0

            hit_candidate_idx = -1
            if prev_action_candidates is not None:
                matches = torch.all(prev_action_candidates == gt_action, dim=1)
                if matches.any():
                    hit_candidate_idx = torch.where(matches)[0][0].item()
                    
            # print(f"[DEBUG] new_tokens_spec before concat: {new_tokens_spec}")
                    
            new_tokens_spec = [torch.cat(new_tokens_spec[i], dim=0) if new_tokens_spec[i] else torch.tensor([], device="cuda") for i in range(num_candidates)]
            new_tokens_main = torch.cat(new_tokens_main, dim=0) if new_tokens_main else torch.tensor([], device="cuda")
            all_tokens_main.append(new_tokens_main.tolist()) if new_tokens_main.numel() > 0 else None

            if hit_candidate_idx != -1:
                # === HIT: Prediction Correct ===
                # 1. Copy KV Cache from Spec Candidate to Main
                print(f"[HINT] ACTION PREDICTION HIT!!!")
                start_time = time.perf_counter()
                # The Spec stream already processed action_T+1 and frame T+2
                # action_start_pos for this verification is the position of action_T+1
                action_start_pos = promptlen + imagenum_main * PERIOD - actnum
                draft_pos = torch.arange(action_start_pos, action_start_pos + actnum + pixnum, device="cuda")
                model.restore_kv_cache(hit_candidate_idx + 1, draft_pos)
                end_time = time.perf_counter()
                print(f"[HINT] KV Cache Restoration Time: {end_time - start_time:.6f} seconds")
                

                # 2. Advance Main State variables (Jump ahead)
                # We assume Spec has generated 'pixnum' tokens. We append them to Main.
               
                imagenum_main += 1
                current_global_pos += (actnum + pixnum) # Jump Action + Frame
                
                # Main is now at end of T+1, ready for T+2
                state_0_len += pixnum 
                # Reset Spec for T+2
                spec_active = False 
                state_1_len = 0
                current_step = 0
                all_tokens_main.append(new_tokens_spec[hit_candidate_idx].tolist()) if new_tokens_spec[hit_candidate_idx].numel() > 0 else None
                
                new_tokens_main = []
                new_tokens_spec = [[] for _ in range(num_candidates)]
                
                prev_action_candidates = None
                
                # Restart Main stream for the next frame
                state_row_lists[0].append(0)
                state_row_tokens_lists[0].zero_()
                ptr_heads[0] = 0
                frame_completed = False
                
                continue # Skip decode step, loop back to check next boundary
                
            else:
                # === MISS: Prediction Wrong ===
                # 1. Inject GT Action into Main (Correction)
                # Main is at end of T. Input GT Action.
                action_start_pos = promptlen + imagenum_main * PERIOD - actnum
                action_input = gt_action
                imagenum_main += 1
                
                if action_input.dim() == 1:
                    action_input = action_input.unsqueeze(0)
                
                # action_pos = torch.arange(action_start_pos, action_start_pos + actnum, device="cuda").unsqueeze(0)
                action_pos = torch.arange(action_start_pos, action_start_pos + actnum, device="cuda")
                
                
                
                # 2. Advance variables
                current_global_pos += actnum
                
                # Main is now at START of T+1 (image part)
                # Reset Spec for T+2 (It will restart later after stagger steps)
                spec_active = False
                state_1_len = 0
                current_step = 0
                
                # Prepare NEXT Prediction (for the future T+2, derived from T+1 which Main is about to make)
                # Predict Action T+1 -> T+2
                start_time = time.perf_counter()
                action_history = kwargs["action"][max(0, imagenum_main - action_prev):imagenum_main]
                # Normalize kwargs["action"] to a Python list of tensors
                raw_actions = kwargs.get("action", [])
                if isinstance(raw_actions, torch.Tensor):
                    raw_actions_list = [t for t in raw_actions]  # unbind into list
                else:
                    raw_actions_list = list(raw_actions)
                action_history = raw_actions_list[max(0, imagenum_main - action_prev):imagenum_main]
                # 若长度小于 action_prev，则在前面补齐全 0 的 action（shape=[1, actnum]）
                if len(action_history) < action_prev:
                    need = action_prev - len(action_history)
                    if len(raw_actions_list) > 0:
                        sample = raw_actions_list[0]
                        pad_list = [torch.zeros((actnum,), device=sample.device, dtype=sample.dtype) for _ in range(need)]
                    else:
                        pad_list = [torch.zeros((actnum,), device="cuda", dtype=torch.long) for _ in range(need)]
                    action_history = pad_list + action_history
                    
                hist_tensor = torch.stack(action_history, dim=0).to("cuda")
                
                # print(f"[DEBUG] Predicting Next Action based on history shape: {hist_tensor.shape}")
                next_action_candidates = action_pred_func(hist_tensor)
                prev_action_candidates = next_action_candidates
                                
                end_time = time.perf_counter()
                print(f"[HINT] Action Prediction Time: {end_time - start_time:.6f} seconds")
                
                # print(f"[DEBUG] action_history shapes: {[a.shape for a in action_history]}")
                # print(f"[DEBUG] all_tokens_main[-1] before draft: {all_tokens_main[-1]}")
                
                start_time = time.perf_counter()

                prev_tokens = torch.as_tensor(all_tokens_main[-1], device="cuda")
                print(f"[DEBUG] prev_tokens shape: {prev_tokens.shape}, action_candidates shape: {next_action_candidates.shape}")
                
                draft_input_ids = draft_func(all_tokens_main[-1], next_action_candidates)
                # draft_pos = torch.arange(action_start_pos + actnum, action_start_pos + actnum + pixnum, device="cuda").unsqueeze(0)
                draft_pos = torch.arange(action_start_pos + actnum, action_start_pos + actnum + pixnum, device="cuda")
                
                end_time = time.perf_counter()
                print(f"[HINT] Draft Time: {end_time - start_time:.6f} seconds")

                
                
                K = draft_input_ids.size(0)
                
                start_time = time.perf_counter()
                
                prefill_input = torch.cat([action_input.expand(K, -1).contiguous(), draft_input_ids], dim=1)
                # prefill_pos = torch.cat([action_pos, draft_pos], dim=1)
                prefill_pos = torch.cat([action_pos, draft_pos], dim=0)
                
                # Setup Next Action for Spec Stream (so it knows what action to use for T+1 -> T+2)
                state_1_last_tokens_setup = next_action_candidates[:, -1].view(num_candidates, 1)
                
                prefill_input = torch.cat([prefill_input.expand(num_candidates, -1), next_action_candidates], dim=1)
                # Next action (for T+2→T+3) starts at action_start_pos + actnum + pixnum
                next_action_start = action_start_pos + actnum + pixnum
                prefill_pos = torch.cat([prefill_pos, torch.arange(next_action_start, next_action_start + actnum, device="cuda")], dim=0)
                # print(f"[DEBUG] Speculative Prefill Input Shape: {prefill_input.shape}, Position Shape: {prefill_pos.shape}")

                start_1_time = time.perf_counter()
                
                prefill_func(model, input_ids=prefill_input, position_ids=prefill_pos)
                
                end_time = time.perf_counter()
                print(f"[HINT] Speculative Prefill(include prep) Time: {end_time - start_time:.6f} seconds, (only prefill){end_time - start_1_time:.6f} seconds")
                
            new_tokens_spec = [[] for _ in range(num_candidates)]
            new_tokens_main = []
            
            # Restart Main stream for the next frame (verification done)
            state_row_lists[0].append(0)
            state_row_tokens_lists[0].zero_()
            ptr_heads[0] = 0
            frame_completed = False


        # Checks for Spec Activation
        if not spec_active:
            if current_step >= stagger_steps and imagenum_main < (num_generate_tokens + pixnum - 1) // pixnum:
                if state_1_last_tokens_setup is None:
                    # Fallback until the first speculative action prediction is ready.
                    state_1_last_tokens_setup = torch.zeros((num_candidates, 1), device="cuda", dtype=torch.long)
                spec_active = True
                state_row_tokens_lists[1].zero_()
                state_row_lists[1] = [0]
                state_row_last_tokens[1] = [None for _ in range(rownum)]
                state_1_len = 0
                ptr_heads[1] = 0
                # 关键：让 spec 从 action 的最后一个 token 开始
                state_row_tokens_lists[1][0] += 1
                state_row_last_tokens[1][0] = state_1_last_tokens_setup
                if state_row_tokens_lists[1][0] == windowsize and state_row_lists[1][0] < rownum - 1:
                    state_row_lists[1].append(1)


        # --- Prepare Inputs (Mixed) ---
        # Pass Main state only if main_active_now is True
        main_args = (state_row_lists[0], state_row_tokens_lists[0], state_row_last_tokens[0])
        if not main_active_now:
             main_args = ([], state_row_tokens_lists[0], [None for _ in range(rownum)])

        input_0, pos_0, input_1, pos_1 = speculative_img_diagd_prepare_inputs(
            main_args,
            (state_row_lists[1], state_row_tokens_lists[1], state_row_last_tokens[1]) if spec_active else ([], state_row_tokens_lists[1], [None for _ in range(rownum)]),
            prompt, imagenum_main, num_candidates,
            pixnum, actnum, columnnum, promptlen, 
            **kwargs
        )

        len0 = input_0.shape[1]
        len1 = input_1.shape[1] if spec_active else 0
        
        # 仅在需要时扩展 token 维（列数可为0也支持）
        if input_0.shape[0] == 1:
            input_0 = input_0.expand(input_1.shape[0], -1)
        # # pos_0 也需要按 batch 扩展
        # if pos_0.shape[0] == 1:
        #     pos_0 = pos_0.expand(pos_1.shape[0], -1)

        packed_input = torch.cat([input_0, input_1], dim=1)
        # packed_pos = torch.cat([pos_0, pos_1], dim=1)
        packed_pos = torch.cat([pos_0, pos_1], dim=0)
        
        # print(f"[DEBUG] Packed Input Shape: {packed_input.shape}, Packed Pos Shape: {packed_pos.shape}")
        # print(f"[DEBUG] Packed Pos: {packed_pos}")
        
        next_tokens_0, next_tokens_1 = None, None
        start_time = time.perf_counter()
        
        if packed_input.shape[1] > 0:
            packed_next_tokens = decode_some_token_function(
                model,
                input_ids=packed_input,
                position_ids=packed_pos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
                
            
            if len0 > 0:
                next_tokens_0 = packed_next_tokens[:, :len0]
                next_tokens_1 = packed_next_tokens[:, len0:]
            else:
                next_tokens_0 = None
                next_tokens_1 = packed_next_tokens

        # print(f"[DEBUG] Packed Next Tokens Shape: {packed_next_tokens.shape}")
            
        current_step += 1 

        def _update_stream_state(idx, schedule, next_tokens_stream, result_accum, last_tokens_list):
            if next_tokens_stream is None:
                return 0

            total_steps = next_tokens_stream.shape[1]
            plan_len = schedule["length"]

            ptr = ptr_heads[idx]
            steps = min(total_steps, plan_len - ptr)
            if steps <= 0:
                return 0
            counts_after = schedule["prefix_counts"][:, ptr + steps]
            state_row_tokens_lists[idx][:] = counts_after

            working_stream = next_tokens_stream[:, :steps, ...]

            # Compute row_ids and stream_chunks before using them below
            plan_rows = schedule["plan_rows"]
            row_ids = plan_rows[ptr : ptr + steps]

            if working_stream.dim() > 2:
                stream_chunks = working_stream.permute(1, 0, *range(2, working_stream.dim()))
            else:
                stream_chunks = working_stream.permute(1, 0)

            if idx == 0:
                # Insert tokens at correct row-major positions (matching img_diagd_decode_n_tokens)
                for offset, row_id in enumerate(row_ids):
                    chunk = stream_chunks[offset]
                    chunk = chunk[0].reshape(1)  # Main's token only, shape [1]
                    # Compute row-major position from prefix_counts.
                    # counts_after gives the global per-row counts INCLUDING the
                    # prefill token, but result_accum (new_tokens_main) starts
                    # empty for each frame, so we subtract 1 to get the
                    # 0-indexed insert position within the current frame.
                    row_pos = counts_after[row_id].item() - 1  # 0-indexed within row
                    prev_total = counts_after[:row_id].sum().item()  # tokens in earlier rows
                    insert_pos = prev_total + row_pos - 1  # subtract 1 for prefill offset
                    result_accum.insert(insert_pos, chunk.clone())
            else:
                for cand_idx, cand_tokens in enumerate(working_stream.unbind(0)):
                    if cand_idx >= len(result_accum):
                        result_accum.append([])
                    result_accum[cand_idx].extend([tok.view(1).clone() for tok in cand_tokens.unbind(0)])

            # Update last_tokens_list for all rows processed in this step
            for offset, row_id in enumerate(row_ids):
                chunk = stream_chunks[offset]
                if idx == 0:
                    chunk = chunk[:1].contiguous()
                last_tokens_list[row_id] = chunk

            rows_after = schedule["rows_state"][ptr + steps]
            rows_ref = state_row_lists[idx]
            rows_ref[:] = list(rows_after)

            if rows_after:
                keep = set(rows_after)
                for r in range(len(last_tokens_list)):
                    if r not in keep:
                        last_tokens_list[r] = None
            else:
                for r in range(len(last_tokens_list)):
                    last_tokens_list[r] = None

            ptr_heads[idx] = ptr + steps
            return steps

        # --- Update States ---
        
        # 1. Update Main State
        start_time = time.perf_counter()
        if main_active_now:
            schedule_main = diag_schedules[0]
            prev_0_len = state_0_len
            added_len = _update_stream_state(0, schedule_main, next_tokens_0, new_tokens_main, state_row_last_tokens[0])
            state_0_len += added_len
            # Sync Main -> Spec cache only when a frame just completed
            # (only_previous mask means intra-frame sync is unnecessary)
            if prev_0_len // pixnum != state_0_len // pixnum and len0 > 0:
                model.restore_kv_cache(0, pos_0)

        # 2. Update Spec State (按 img_diagd_decode_n_tokens 逻辑)
        if next_tokens_1 is not None and next_tokens_1.shape[1] > 0:
            if len(state_row_lists[1]) == 0:
                state_1_len += 1
                last_tok = next_tokens_1[:, -1].clone()
                state_row_last_tokens[1][0] = last_tok
                for cand_idx, tok in enumerate(last_tok.unbind(0)):
                    if cand_idx >= len(new_tokens_spec):
                        new_tokens_spec.append([])
                    new_tokens_spec[cand_idx].append(tok.view(1).clone())
                state_row_lists[1].append(0)
                state_row_tokens_lists[1][0] += 1
                if state_row_tokens_lists[1][0] == windowsize and state_row_lists[1][0] < rownum - 1:
                    state_row_lists[1].append(1)
            else:
                need_remove_row = None
                num_new_tokens = next_tokens_1.shape[1]
                state_1_len += num_new_tokens

                for i in range(num_new_tokens):
                    row_id = state_row_lists[1][i]
                    position_in_new_tokens = torch.sum(state_row_tokens_lists[1][:(row_id + 1)], dim=0) + imagenum_main * pixnum
                    pos_ins = int(position_in_new_tokens.item()) if torch.is_tensor(position_in_new_tokens) else int(position_in_new_tokens)

                    # 按候选逐一插入
                    for cand_idx in range(next_tokens_1.shape[0]):
                        if cand_idx >= len(new_tokens_spec):
                            new_tokens_spec.append([])
                        new_tokens_spec[cand_idx].insert(pos_ins, next_tokens_1[cand_idx, i].view(1).clone())

                    state_row_last_tokens[1][row_id] = next_tokens_1[:, i].clone()
                    state_row_tokens_lists[1][row_id] += 1

                    if state_row_tokens_lists[1][row_id] == windowsize and state_row_lists[1][i] < rownum - 1:
                        state_row_lists[1].append(state_row_lists[1][i] + 1)

                    elif state_row_lists[1][i] == rownum - 1 and state_row_tokens_lists[1][row_id] == columnnum:
                        state_row_tokens_lists[1] = torch.zeros((rownum,), dtype=torch.long, device="cuda")
                        state_row_lists[1] = []
                        state_row_last_tokens[1] = [None for _ in range(rownum)]
                        need_remove_row = None
                        break

                    if state_row_tokens_lists[1][row_id] == columnnum: ## this row is done
                        need_remove_row = state_row_lists[1][i]

                if need_remove_row is not None:
                    state_row_lists[1].remove(need_remove_row)
                    state_row_last_tokens[1][need_remove_row] = None
             
        if len(state_row_lists[0]) == 0 and state_0_len < num_generate_tokens:
             # A frame just completed. Flush its tokens as a COMPLETE 336-token
             # frame (the prefill token is stored separately in all_tokens_main[1],
             # so prepend it to the 335 tokens accumulated in new_tokens_main).
             if new_tokens_main:
                 frame_tokens = torch.cat(new_tokens_main, dim=0).tolist()
                 if len(frame_tokens) == pixnum - 1 and first_pixel_token is not None:
                     frame_tokens = [int(first_pixel_token.squeeze().item())] + frame_tokens
                 all_tokens_main.append(frame_tokens)
                 new_tokens_main = []
             frame_completed = True
             # Do NOT restart here — restart happens after verification at loop top.

    return all_tokens_main[1:]

DIAG_SCHEDULE_CACHE: Dict[Tuple[int, int], Dict[str, Any]] = {}

def _build_diag_schedule(rownum: int, columnnum: int, windowsize: int) -> Dict[str, Any]:

    rows_ref: List[int] = [0]
    row_cnts = [0 for _ in range(rownum)]
    plan_rows: List[int] = []
    rows_state: List[Tuple[int, ...]] = [tuple(rows_ref)]  # t=0

    while rows_ref:
        snapshot = list(rows_ref)
        for row_id in snapshot:
            plan_rows.append(row_id)
            row_cnts[row_id] += 1

            if (
                row_cnts[row_id] == windowsize
                and row_id < rownum - 1
                and (row_id + 1) not in rows_ref
            ):
                rows_ref.append(row_id + 1)

            if row_cnts[row_id] >= columnnum and row_id in rows_ref:
                rows_ref.remove(row_id)

            # 每次物理写入后快照，行数对齐 plan_rows
            rows_state.append(tuple(rows_ref))

    plan_tuple = tuple(plan_rows)
    prefix_counts = torch.zeros((rownum, len(plan_tuple) + 1), dtype=torch.int64)
    for pos, row_id in enumerate(plan_tuple, start=1):
        prefix_counts[:, pos] = prefix_counts[:, pos - 1]
        prefix_counts[row_id, pos] += 1

    return {
        "plan_rows": plan_tuple,
        "rows_state": tuple(rows_state),  # 长度 = len(plan_rows) + 1
        "prefix_counts": prefix_counts.to("cuda"),
        "length": len(plan_tuple),
    }

def _get_diag_schedule(rownum: int, columnnum: int, windowsize: int, frame_idx: int = 0) -> Dict[str, Any]:
    key = (windowsize, frame_idx)
    schedule = DIAG_SCHEDULE_CACHE.get(key)
    if schedule is None:
        print(f"[DAMN] schedule is None!")
        schedule = _build_diag_schedule(rownum, columnnum, windowsize)
        DIAG_SCHEDULE_CACHE[key] = schedule
    return schedule