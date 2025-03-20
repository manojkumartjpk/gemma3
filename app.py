import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from typing import Optional
import inferless
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="Describe this image in detail.")
    image_url: Optional[str] = None
    system_prompt: Optional[str] = "You are a helpful assistant."
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.1
    repetition_penalty: Optional[float] = 1.18
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 100
    do_sample: Optional[bool] = False

@inferless.response
class ResponseObjects(BaseModel):
    generated_text: str = Field(default="Test output")

class InferlessPythonModel:
    def initialize(self):
        model_id = "google/gemma-3-4b-it"
        snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="cuda"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def infer(self, request: RequestObjects) -> ResponseObjects:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": request.system_prompt}]
            }
        ]
        # Build the user message based on provided inputs.
        user_content = []
        if request.image_url is not None:
            user_content.append({"type": "image", "image": request.image_url})
        
        user_content.append({"type": "text", "text": request.prompt})
        messages.append({
            "role": "user",
            "content": user_content
        })

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.do_sample,
                repetition_penalty=request.repetition_penalty,
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return ResponseObjects(generated_text=decoded)

    def finalize(self):
        self.model = None
