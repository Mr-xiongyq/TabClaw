import base64
import mimetypes
import re
from openai import AsyncOpenAI
from typing import List, Dict, Optional, AsyncGenerator


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        model_extra_params: dict = None,
        vision_model: Optional[str] = None,
        vision_model_extra_params: Optional[dict] = None,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.model_extra_params = model_extra_params
        self.vision_model = vision_model or model
        self.vision_model_extra_params = vision_model_extra_params

    async def chat(self, messages: List[Dict], tools: Optional[List] = None) -> object:
        """Non-streaming chat, returns the message object."""
        kwargs = dict(model=self.model, extra_body=self.model_extra_params, messages=messages, temperature=0.1)
        if tools:
            kwargs["tools"] = tools
        resp = await self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message

    async def stream_chat(self, messages: List[Dict], tools: Optional[List] = None) -> AsyncGenerator:
        """Streaming chat, yields raw chunks."""
        kwargs = dict(model=self.model, extra_body=self.model_extra_params, messages=messages, temperature=0.1, stream=True)
        if tools:
            kwargs["tools"] = tools
        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            yield chunk

    async def image_to_html(self, image_bytes: bytes, filename: str) -> str:
        """Use a multimodal model to transcribe a table image into HTML."""
        mime_type = mimetypes.guess_type(filename)[0] or "image/png"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{image_b64}"

        prompt = (
            "You are converting an image into faithful HTML for downstream table reasoning.\n\n"
            "Requirements:\n"
            "- Output ONLY HTML, no markdown fences, no explanation.\n"
            "- Preserve every visible table exactly as structured in the image.\n"
            "- Use <table>, <thead>, <tbody>, <tr>, <th>, and <td> where appropriate.\n"
            "- Preserve merged cells with rowspan/colspan when visible.\n"
            "- If the image contains multiple separate tables, include all of them in one HTML document.\n"
            "- If there is surrounding title or note text that clarifies the table, keep it above the relevant table.\n"
            "- Do not invent missing values.\n"
            "- Return a valid HTML fragment or document that pandas can parse."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        resp = await self.client.chat.completions.create(
            model=self.vision_model,
            extra_body=self.vision_model_extra_params,
            messages=messages,
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        content = re.sub(r"^```(?:html)?\s*", "", content.strip(), flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content.strip())
        return content.strip()
