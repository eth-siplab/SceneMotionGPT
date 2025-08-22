import random
import numpy as np
from .dataset_t2m import Text2MotionDataset

import logging
import os
import traceback

import rich
import random
import pickle
import codecs as cs
import numpy as np
from rich.progress import track
from os.path import join as pjoin
import json
from torch.utils import data



class Text2MotionDatasetEval(data.Dataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        w_vectorizer,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        code_path='VQVAE',
        task_path=None,
        std_text=False,
        **kwargs,
    ):
        self.min_length = 20
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length

        self.w_vectorizer = w_vectorizer

        # Data mean and std
        self.mean = mean
        self.std = std

        # Data path
        split_file = pjoin(data_root, split + '.txt')
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        motion_token_dir = pjoin(data_root, code_path)

        text_dir = pjoin(data_root, 'texts')

        # Data id list
        self.id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        # Debug mode
        if tiny or debug:
            enumerator = enumerate(self.id_list)
            maxdata = 100
            subset = '_tiny'
        else:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading {data_root} {split}",
                ))
            maxdata = 1e10
            subset = ''

        new_name_list = []
        length_list = []
        data_dict = {}

        # Fast loading
        if os.path.exists(pjoin(data_root, f'tmp/{split}{subset}_data.pkl')):
            if tiny or debug:
                with open(pjoin(data_root, f'tmp/{split}{subset}_data.pkl'),
                          'rb') as file:
                    data_dict = pickle.load(file)
            else:
                with rich.progress.open(
                        pjoin(data_root, f'tmp/{split}{subset}_data.pkl'),
                        'rb',
                        description=f"Loading {data_root} {split}") as file:
                    data_dict = pickle.load(file)
            with open(pjoin(data_root, f'tmp/{split}{subset}_index.pkl'),
                      'rb') as file:
                name_list = pickle.load(file)
            for name in new_name_list:
                length_list.append(data_dict[name]['length'])

        else:
            for idx, name in enumerator:
                if len(new_name_list) > maxdata:
                    break
                try:
                    motion = np.load(pjoin(motion_dir, name + ".npy"))
                    m_token_list = np.load(pjoin(motion_token_dir, f'{name}.npy'))
                    if (len(motion)) < self.min_motion_length:
                        continue

                    # Read text
                    text_data = []
                    flag = False
                    valid_annotation_found = False
                    with cs.open(pjoin(text_dir, name + '.txt')) as f:
                        lines = f.readlines()
                        subseq_counter = 0
                        for line in lines:
                            try:
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                t_tokens = line_split[1].split(' ')
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict['caption'] = caption
                                text_dict['tokens'] = t_tokens
                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                    valid_annotation_found = True
                                else:
                                    motion_new = motion[int(f_tag *
                                                            fps):int(to_tag * fps)]
                                    if (len(motion_new)
                                    ) < self.min_motion_length or (
                                            len(motion_new) >= 200):
                                        continue

                                    m_token_list_new = [
                                        tokens[int(f_tag * fps / unit_length
                                                   ):int(to_tag * fps /
                                                         unit_length)]
                                        for tokens in m_token_list
                                        if int(f_tag * fps / unit_length) <
                                           int(to_tag * fps / unit_length)
                                    ]

                                    if len(m_token_list_new) == 0:
                                        continue

                                    if f_tag < 0 or to_tag < 0 or (to_tag <= f_tag):
                                        raise ValueError(
                                            f"Invalid tags for {name}: f_tag={f_tag}, to_tag={to_tag}"
                                        )
                                    new_name = f"{name}_subseq_{subseq_counter}"
                                    while new_name in data_dict:
                                        subseq_counter += 1
                                        new_name = f"{name}_subseq_{subseq_counter}"
                                    if new_name in data_dict:
                                        logging.warning(
                                            f"Duplicate subsequence name {new_name} found, overwriting."
                                        )

                                    data_dict[new_name] = {
                                        'motion': motion_new,
                                        "length": len(motion_new),
                                        'm_token_list': m_token_list_new,
                                        'text': [text_dict]
                                    }
                                    new_name_list.append(new_name)
                                    length_list.append(len(motion_new))
                                    subseq_counter += 1  # Increment for next subsequence
                                    valid_annotation_found = True
                            except ValueError as e:
                                logging.error(f"Error processing {name}: {e}")
                                continue  # Skip this line if there's an error

                    if not valid_annotation_found:
                        continue  # skip only if no valid annotation

                    if flag:
                        data_dict[name] = {
                            'motion': motion,
                            "length": len(motion),
                            'm_token_list': m_token_list,
                            'text': text_data
                        }
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e:
                    print(f"Error loading {name} in {split} split: {e}")

                    traceback.print_exc()

            name_list, length_list = zip(
                *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

            if tmpFile:
                os.makedirs(pjoin(data_root, 'tmp'), exist_ok=True)
                with open(pjoin(data_root, f'tmp/{split}{subset}_data.pkl'),
                          'wb') as file:
                    pickle.dump(data_dict, file)
                with open(pjoin(data_root, f'tmp/{split}{subset}_index.pkl'),
                          'wb') as file:
                    pickle.dump(name_list, file)

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.nfeats = data_dict[name_list[0]]['motion'].shape[1]
        self.reset_min_len(self.min_length)


        if task_path:
            instructions = task_path
        elif stage == 'lm_pretrain':
            instructions = pjoin(data_root, 'template_pretrain.json')
        elif stage in ['lm_instruct', "lm_rl"]:
            instructions = pjoin(data_root, 'template_instructions.json')
        else:
            raise NotImplementedError(f"stage {stage} not implemented")
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.min_length = length

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        # task = np.random.choice(self.tasks)
        # Get text data
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, m_token_list, text_list = data["motion"], data["length"], data["m_token_list"], data["text"]

        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        # Randomly select a caption
        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        tasks =random.choice(self.tasks)

        # Text
        max_text_len = 20
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        
        # Random crop
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        coin = np.random.choice([False, False, True])
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]

        m_tokens_len = m_tokens.shape[0]
        
        # Z Normalization
        motion = (motion - self.mean) / self.std

        return {
            "text": caption,
            "motion": motion,
            "motion_len": m_length,
            "motion_tokens": m_tokens,
            "motion_tokens_len": m_tokens_len,
            "word_embs": word_embeddings,
            "pos_ohot": pos_one_hots,
            "text_len": sent_len,
            "tokens": "_".join(tokens),
            "all_captions": all_captions,
            "tasks": tasks
        }
