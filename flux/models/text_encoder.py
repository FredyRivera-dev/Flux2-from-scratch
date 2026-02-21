import torch
import torch.nn as nn
from einops import rearrange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512

class Qwen3Embedder(nn.Module):
    def __init__(
        self,
        model_spec: str,
        device: str | torch.device = "cuda",
    ):
        super().__init__()


        # I'm going to use flash_attetion and bfloat16 to make inference faster.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            torch_dtype=torch.bfloat16,
            device_map=str(device),
            attn_implementation="flash_attention_2",
        )
        
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_spec)
        self.max_length = MAX_LENGTH

    @torch.no_grad()
    def forward(self, txt: list[str]):
        ## Btw, this is more efficient and faster.
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in txt
        ]

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)

        output = self.model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack(
            [output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3],
            dim=1,
        )

        return rearrange(out, "b c l d -> b l (c d)")

    def test_txt(self, txt: str) -> bool:
        raise NotImplementedError("Qwen3Embedder does not support text testing")

    def test_image(self, image) -> bool:
        raise NotImplementedError("Qwen3Embedder does not support image testing")

    def upsample_prompt(self, txt: list[str], img=None, **kwargs) -> list[str]:
        raise NotImplementedError("Qwen3Embedder does not support upsampling")


def load_qwen3_embedder(variant: str, device: str | torch.device = "cuda"):
    return Qwen3Embedder(model_spec=f"Qwen/Qwen3-{variant}", device=device)