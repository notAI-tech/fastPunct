import os
import torch
import pydload
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_URLS = {
    "english": {
        "pytorch_model.bin": "https://github.com/notAI-tech/fastPunct/releases/download/v2/pytorch_model.bin",
        "config.json": "https://github.com/notAI-tech/fastPunct/releases/download/v2/config.json",
        "special_tokens_map.json": "https://github.com/notAI-tech/fastPunct/releases/download/v2/special_tokens_map.json",
        "spiece.model": "https://github.com/notAI-tech/fastPunct/releases/download/v2/spiece.model",
        "tokenizer_config.json": "https://github.com/notAI-tech/fastPunct/releases/download/v2/tokenizer_config.json",
    },
}


class FastPunct:
    tokenizer = None
    model = None

    def __init__(self, language='english', checkpoint_local_path=None):

        model_name = language.lower()

        if model_name not in MODEL_URLS:
            print(f"model_name should be one of {list(MODEL_URLS.keys())}")
            return None

        home = os.path.expanduser("~")
        lang_path = os.path.join(home, ".FastPunct_" + model_name)

        if checkpoint_local_path:
            lang_path = checkpoint_local_path

        if not os.path.exists(lang_path):
            os.mkdir(lang_path)

        for file_name, url in MODEL_URLS[model_name].items():
            file_path = os.path.join(lang_path, file_name)
            if os.path.exists(file_path):
                continue
            print(f"Downloading {file_name}")
            pydload.dload(url=url, save_to_path=file_path, max_time=None)

        self.tokenizer = T5Tokenizer.from_pretrained(lang_path)
        self.model = T5ForConditionalGeneration.from_pretrained(
            lang_path, return_dict=True
        )

        if torch.cuda.is_available():
            print(f"Using GPU")
            self.model = self.model.cuda()

    def punct(
        self, sentences, beam_size=1, max_len=None, correct=False
    ):
        return_single = True
        if isinstance(sentences, list):
            return_single = False
        else:
            sentences = [sentences]

        prefix = 'punctuate'
        if correct:
            beam_size = 8
            prefix = 'correct'

        input_ids = self.tokenizer(
            [
                f"{prefix}: {sentence}"
                for sentence in sentences
            ],
            return_tensors="pt",
            padding=True,
        ).input_ids

        if not max_len:
            max_len = max([len(tokenized_input) for tokenized_input in input_ids]) + max([len(s.split()) for s in sentences]) + 4

        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        output_ids = self.model.generate(
            input_ids, num_beams=beam_size, max_length=max_len
        )

        outputs = [
            self.tokenizer.decode(output_id, skip_special_tokens=True)
            for output_id in output_ids
        ]

        if return_single:
            outputs = outputs[0]

        return outputs
