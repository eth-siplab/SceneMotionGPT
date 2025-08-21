import torch
import rich
import pickle
import numpy as np


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# padding to max length in one batch
def collate_tensors(batch):
    if isinstance(batch[0], np.ndarray):
        batch = [torch.tensor(b).float() for b in batch]

    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def humanml3d_collate(batch):
    """
    Collate function for dictionary-based dataset returns.
    """
    notnone_batches = [b for b in batch if b is not None]

    # Check if this is an evaluation batch by looking for word_embs
    EvalFlag = "word_embs" in notnone_batches[0]

    # Sort by text length for evaluation batches
    if EvalFlag:
        notnone_batches.sort(key=lambda x: x["text_len"], reverse=True)

    adapted_batch = {}
    # Motion data (always present)
    if "motion" in notnone_batches[0]:
        adapted_batch.update({
            "motion": collate_tensors([torch.tensor(b["motion"]).float() for b in notnone_batches]),
            "motion_len": [b["motion_len"] for b in notnone_batches],
        })

    if "motion_tokens" in notnone_batches[0]:
        adapted_batch.update({
            "motion_tokens": collate_tensors([torch.tensor(b["motion_tokens"]).float() for b in notnone_batches]),
            "motion_tokens_len": [b["motion_tokens_len"] for b in notnone_batches],
        })

    # Text data (always present)
    if "text" in notnone_batches[0]:
        adapted_batch.update({
            "text": [b["text"] for b in notnone_batches],
            "all_captions": [b["all_captions"] for b in notnone_batches],
        })

    if "name" in notnone_batches[0]:
        adapted_batch.update({
            "name": [b["name"] for b in notnone_batches],
        })

    # Evaluation fields
    if EvalFlag:
        adapted_batch.update({
            "word_embs": collate_tensors([torch.tensor(b["word_embs"]).float() for b in notnone_batches]),
            "pos_ohot": collate_tensors([torch.tensor(b["pos_ohot"]).float() for b in notnone_batches]),
            "text_len": collate_tensors([torch.tensor(b["text_len"]) for b in notnone_batches]),
            "tokens": [b["tokens"] for b in notnone_batches],
        })

    # Tasks (if present)
    if "tasks" in notnone_batches[0]:
        adapted_batch.update({"tasks": [b["tasks"] for b in notnone_batches]})

    return adapted_batch


def load_pkl(path, description=None, progressBar=False):
    if progressBar:
        with rich.progress.open(path, 'rb', description=description) as file:
            data = pickle.load(file)
    else:
        with open(path, 'rb') as file:
            data = pickle.load(file)
    return data
