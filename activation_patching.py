import sys
import numpy as np
import torch
from torch.nn.functional import log_softmax
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from tqdm import tqdm

model_dir = '/home/leiyu/scratch/cma_hallucination/models/gpt2-xl/'
data_dir = '/home/leiyu/scratch/cma_hallucination/data/'

torch.set_grad_enabled(False)
num_layers = 48
model_hidden_dim = 1600
n_intervene = 10
n_hall_sample = 1000
module_kinds = ['res', 'attn', 'mlp']


def untuple(x):
    return x[0] if isinstance(x, tuple) else x


def get_key_pos_effects(effects, row):
    # effects: (n_layer, seq_len)
    effects_subj_first = effects[:, row['cue entity start idx']]
    if row['cue entity end idx'] > row['cue entity start idx'] + 1:
        effects_subj_mid = effects[:, 1 + row['cue entity start idx']:row['cue entity end idx']].mean(-1)
    else:
        effects_subj_mid = effects_subj_first
    effects_subj_last = effects[:, row['cue entity end idx']]

    effects_first_after = effects[:, 1 + row['cue entity end idx']]
    effects_further = effects[:, 1 + row['cue entity end idx']:-1].mean(-1)
    effects_last = effects[:, -1]

    igs_key_pos = torch.stack([
        effects_subj_first, effects_subj_mid, effects_subj_last, effects_first_after, effects_further, effects_last
    ], -1)

    return igs_key_pos  # (n_layer, 6)


def run_get_noise_and_te(model, tokenizer, prompt, true_obj_id, hall_obj_id,
                         subj_start, subj_end,
                         batch_size=100, intervene_ent='subj'):
    batch_inputs = tokenizer(
        [prompt] * (1 + batch_size), return_tensors='pt'
    )['input_ids'].to(model.device)

    def make_wte_noise_hook(noise, intervene_start, intervene_end):

        def wte_noise_hook(module, inputs, outputs):
            outputs_0 = untuple(outputs)
            outputs_0[1:, intervene_start:intervene_end] += noise.to(outputs_0.device)
            return outputs

        return wte_noise_hook

    if intervene_ent == 'subj':
        intervene_start, intervene_end = subj_start, subj_end + 1
    else:
        intervene_start, intervene_end = subj_end + 1, batch_inputs.shape[1] - 1

    noise = torch.randn(batch_size, 1, model_hidden_dim)
    emb_hook = model.transformer.wte.register_forward_hook(
        make_wte_noise_hook(noise, intervene_start, intervene_end)
    )
    with torch.no_grad():
        logits = model(batch_inputs.to(model.device)).logits[:, -1]  # (B, vocab_size)
        obj_logit_diffs = (logits[:, hall_obj_id] - logits[:, true_obj_id]).cpu()

    valid_batch_idx = (obj_logit_diffs[1:] < obj_logit_diffs[0]).nonzero(as_tuple=True)[0]
    # print('valid_batch_idx: ', valid_batch_idx)
    n_valid_noise = len(valid_batch_idx)
    valid_noise_rate = float(n_valid_noise) / batch_size

    if 1 < len(valid_batch_idx) < n_intervene:
        valid_batch_idx = torch.cat([
            valid_batch_idx, valid_batch_idx[-1].repeat(n_intervene - len(valid_batch_idx))
        ])
    elif len(valid_batch_idx) > n_intervene:
        valid_batch_idx = valid_batch_idx[:n_intervene]
    else:
        valid_batch_idx = torch.arange(n_intervene)

    if len(valid_batch_idx) != n_intervene:
        print(len(valid_batch_idx))

    emb_noises = noise[valid_batch_idx]
    TEs = obj_logit_diffs[1:][valid_batch_idx] - obj_logit_diffs[0]

    emb_hook.remove()
    torch.cuda.empty_cache()

    return emb_noises, TEs, obj_logit_diffs[0], valid_noise_rate


def run_with_activation_patch(model, batch_inputs, batch_noise,
                              intervene_start, intervene_end,
                              true_obj_id, hall_obj_id,
                              patch_layer_idx, patch_seq_idx,
                              module_kind='res'):
    hooks = []

    def make_wte_noise_hook(noise, intervene_start, intervene_end):

        def wte_noise_hook(module, inputs, outputs):
            outputs_0 = untuple(outputs)
            outputs_0[1:, intervene_start:intervene_end] += noise.to(outputs_0.device).unsqueeze(0)
            return outputs

        return wte_noise_hook

    emb_hook = model.transformer.wte.register_forward_hook(
        make_wte_noise_hook(batch_noise, intervene_start, intervene_end)
    )
    hooks.append(emb_hook)

    # Define the model-patching hook for computing the indirect effects
    for i in range(len(patch_layer_idx)):

        def make_patching_hook(patched_batch_id, patched_seq_id):
            def patching_hook(module, inputs, outputs):
                outputs_0 = untuple(outputs)  # (B, seq_len, hidden_dim)
                outputs_0[patched_batch_id, patched_seq_id] = outputs_0[0, patched_seq_id]

                return outputs

            return patching_hook

        if module_kind == 'res':
            hook_i = model.transformer.h[patch_layer_idx[i]].register_forward_hook(
                make_patching_hook(i + 2, patch_seq_idx[i]))
            hooks.append(hook_i)
        elif module_kind == 'attn':
            patch_layer_start = max(0, patch_layer_idx[i] - 5)
            patch_layer_end = min(patch_layer_idx[i] + 5, num_layers)
            for j in range(patch_layer_start, patch_layer_end):
                hook_ij = model.transformer.h[j].attn.register_forward_hook(
                    make_patching_hook(i + 2, patch_seq_idx[i]))
                hooks.append(hook_ij)
        elif module_kind == 'mlp':
            patch_layer_start = max(0, patch_layer_idx[i] - 5)
            patch_layer_end = min(patch_layer_idx[i] + 5, num_layers)
            for j in range(patch_layer_start, patch_layer_end):
                hook_ij = model.transformer.h[j].mlp.register_forward_hook(
                    make_patching_hook(i + 2, patch_seq_idx[i]))
                hooks.append(hook_ij)
        else:
            raise ValueError('Invalid patching module kind')

    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad():
        batch_log_probs = log_softmax(model(batch_inputs).logits[:, -1], -1).cpu()  # (B, vocab_size)
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()

    log_prob_diffs = batch_log_probs[:, hall_obj_id] - batch_log_probs[:, true_obj_id]

    return log_prob_diffs[2:] - log_prob_diffs[1]


def main(module_kind):
    device = torch.device('cuda')
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    pararel_df = pd.read_csv(data_dir + 'dataframes/pararel_questions.csv')

    # sampled_hall_idx = np.load(data_dir + 'results/pararel_sampled_hall_idx_act.npy')
    sampled_hall_idx = pararel_df.loc[
        pararel_df['is hallucination'] == 1
    ].sample(n_hall_sample).index
    pararel_hall_df = pararel_df.iloc[sampled_hall_idx].reset_index(drop=True)
    np.save(data_dir + 'results/pararel_sampled_hall_idx_act_{}.npy'.format(module_kind), np.array(sampled_hall_idx))

    results = {
        'TE': [], 'IE': [],
        'y_0': [], 'noise': []
    }
    valid_noise_rate = []
    for i, row in tqdm(pararel_hall_df.iterrows(), total=pararel_hall_df.shape[0]):
        seq_len = len(tokenizer(row['prompt'])['input_ids'])
        true_obj_id, hall_obj_id = row['true object first token id'], row['predicted object first token id']
        subj_start, subj_end = row['cue entity start idx'], row['cue entity end idx']
        batch_size = seq_len * num_layers + 2
        batch_inputs = tokenizer(
            [row['prompt']] * batch_size, return_tensors='pt'
        )['input_ids'].to(device)

        noise_i, TE, y_0, val_noise_r = run_get_noise_and_te(
            model, tokenizer, row['prompt'], true_obj_id, hall_obj_id,
            subj_start, subj_end, intervene_ent='subj')

        results['y_0'].append(y_0)  # (1,)
        results['TE'].append(TE.mean())  # (1,)
        results['noise'].append(noise_i)  # (n_intervene, 1, 1600)
        valid_noise_rate.append(val_noise_r)

        patch_layer_idx = torch.arange(num_layers).repeat_interleave(seq_len)
        patch_seq_idx = torch.arange(seq_len).repeat(num_layers)

        IEs = []
        for j in range(n_intervene):
            IEs_ij = run_with_activation_patch(
                model, batch_inputs, noise_i[j],
                subj_start, subj_end+1,
                true_obj_id, hall_obj_id,
                patch_layer_idx, patch_seq_idx,
                module_kind=module_kind
            )
            torch.cuda.empty_cache()

            key_pos_IEs = get_key_pos_effects(IEs_ij.view(num_layers, seq_len), row)
            IEs.append(key_pos_IEs)  # (num_layer, 6)

        results['IE'].append(torch.stack(IEs).mean(0))

    for k, v in results.items():
        torch.save(
            torch.stack(v), data_dir + 'results/pararel/{}_{}_loc.pt'.format(k, module_kind)
        )

    np.save(data_dir + 'results/pararel/valid_noise_rate_{}_loc.npy'.format(module_kind), valid_noise_rate)


if __name__ == '__main__':
    module_kind = module_kinds[int(sys.argv[1])]
    main(module_kind)
    # python activation_patching.py 0
