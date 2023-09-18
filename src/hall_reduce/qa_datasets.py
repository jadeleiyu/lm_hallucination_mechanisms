import json
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer


class MENDQADataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, *args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "zsre_mend_eval.json"
        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            assert (
                "nq question: " in record["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
            ans_toks = tok(" " + record["loc_ans"])["input_ids"]
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        "target_new": {"str": record["answers"][0]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )

        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)


class FactQADataset:
    """
    Dataset for evaluating fact-QA questions from NaturalQA and TriviaQA datasets.
    The MEND, KE and KN baselines can be evaluated on this dataset, while ROME cannot be evaluated
    since it requires an explicit "subject" token, which is not available here.
    """
    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        df_loc = data_dir / "fact_qa.csv"
        df = pd.read_csv(df_loc)

        data = []
        for i, row in df.iterrows():
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": row["question"],
                        "subject": "",
                        "target_new": {"str": row["true answer"]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [],
                    "neighborhood_prompts": [],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )
        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)


