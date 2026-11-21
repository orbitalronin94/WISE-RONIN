import torch

from scripts.evaluate_pilot import fit_prompt, generate_batch

class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids

class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return FakeEncoding(list(range(3, 203)))
    def token_to_id(self, token):
        return {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2}[token]
    def decode(self, ids, skip_special_tokens=True):
        return ""

class FakeTokenizerWrapper:
    tokenizer = FakeTokenizer()

class FakeModel:
    def __init__(self):
        self.seen_widths = []
        self.reset_flags = []
    def __call__(self, batch, reset_state=True):
        self.seen_widths.append(batch.shape[1])
        self.reset_flags.append(reset_state)
        logits = torch.zeros(batch.shape[0], batch.shape[1], 256)
        logits[:, :, 3 if reset_state else 2] = 1.0
        return {"logits": logits}

def test_fit_prompt_preserves_question_head_and_answer_suffix():
    fitted = fit_prompt(list(range(200)), max_context=128)
    assert len(fitted) == 127
    assert fitted[:3] == [0, 1, 2]
    assert fitted[-3:] == [197, 198, 199]

def test_generate_batch_slides_within_shared_128_token_context():
    model = FakeModel()
    outputs = generate_batch(model, ["long question"], FakeTokenizerWrapper(),
                             "lumen", max_new=64, max_context=128)
    assert outputs[0]["text"] == ""
    assert outputs[0]["eos_terminated"] is True
    assert outputs[0]["generated_tokens"] == 2
    assert model.seen_widths
    assert max(model.seen_widths) <= 128
    assert model.seen_widths == [128, 1]
    assert model.reset_flags == [True, False]

def test_lumen_prompts_are_prefilled_individually_to_avoid_padding_state_leaks():
    model = FakeModel()
    outputs = generate_batch(model, ["question one", "question two"],
                             FakeTokenizerWrapper(), "lumen",
                             max_new=64, max_context=128)
    assert [item["text"] for item in outputs] == ["", ""]
    assert all(item["eos_terminated"] for item in outputs)
    assert model.reset_flags == [True, False, True, False]
    assert model.seen_widths == [128, 1, 128, 1]
