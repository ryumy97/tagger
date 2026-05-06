#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import sys
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "google/gemma-4-E4B-it"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a structured JSON object (tag/title/description/rating) "
            "using local transformers inference with google/gemma-4-31b-it."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        help="Path to a single image file (jpg, png, webp, etc.).",
    )
    input_group.add_argument(
        "--dir",
        help="Path to a directory that contains images to process.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Analyze this image and output JSON only. "
            "Do not transcribe or quote any visible text in the image."
        ),
        help="Instruction prompt to guide tagging (default disables OCR-style output).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for model output (default: 0.2).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum generated tokens (default: 300).",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"Transformers model id (default: {MODEL_ID}).",
    )
    return parser.parse_args()


def _extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return text


def _coerce_payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Model response is not a JSON object.")

    tags = data.get("tag", [])
    title = data.get("title", "")
    description = data.get("description", "")
    rating = data.get("rating", 0)

    if not isinstance(tags, list):
        tags = [str(tags)]
    tags = [str(t).strip() for t in tags if str(t).strip()]

    title = str(title).strip()
    description = str(description).strip()

    try:
        rating = int(rating)
    except Exception:
        rating = 0
    rating = max(1, min(10, rating))

    return {
        "tag": tags,
        "title": title,
        "description": description,
        "rating": rating,
    }


def build_messages(prompt: str) -> list[dict[str, Any]]:
    system_prompt = (
        "You are a strict JSON generator. Output ONLY valid JSON.\n"
        "Required schema:\n"
        "{\n"
        '  "tag": string[],\n'
        '  "title": string,\n'
        '  "description": string,\n'
        '  "rating": integer (1-10)\n'
        "}\n"
        "Do not include markdown or extra keys."
    )
    user_prompt = (
        f"{prompt}\n"
        "Return only the required JSON object for image content semantics."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def run_transformers_inference(
    processor: Any,
    model: Any,
    messages: list[dict[str, Any]],
    image_path: str,
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> str:
    if not os.path.isfile(image_path):
        raise ValueError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=prompt_text,
        images=image,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    generated = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
    )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_length:]
    output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return output_text.strip()


def load_model_and_processor(model_id: str) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return processor, model


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def list_images_in_directory(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")
    entries = sorted(os.listdir(directory))
    return [
        os.path.join(directory, name)
        for name in entries
        if os.path.isfile(os.path.join(directory, name)) and is_image_file(name)
    ]


def process_one_image(
    processor: Any,
    model: Any,
    image_path: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    content = run_transformers_inference(
        processor=processor,
        model=model,
        messages=build_messages(prompt),
        image_path=image_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    json_str = _extract_json_block(content)
    parsed = json.loads(json_str)
    return _coerce_payload(parsed)


def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    index = 1
    while True:
        candidate = f"{base}_{index}{ext}"
        if not os.path.exists(candidate):
            return candidate
        index += 1


def save_to_rating_folder(base_dir: str, image_path: str, result: dict[str, Any]) -> dict[str, str]:
    rating_dir = os.path.join(base_dir, str(result["rating"]))
    os.makedirs(rating_dir, exist_ok=True)

    image_name = os.path.basename(image_path)
    target_image = _unique_path(os.path.join(rating_dir, image_name))
    shutil.copy2(image_path, target_image)

    stem, _ = os.path.splitext(os.path.basename(target_image))
    json_path = os.path.join(rating_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True, indent=2)

    return {"image": target_image, "json": json_path}


def main() -> int:
    args = parse_args()

    try:
        processor, model = load_model_and_processor(args.model_id)

        if args.image:
            output = process_one_image(
                processor=processor,
                model=model,
                image_path=args.image,
                prompt=args.prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            print(json.dumps(output, ensure_ascii=True, indent=2))
        else:
            images = list_images_in_directory(args.dir)
            if not images:
                raise ValueError(f"No image files found in directory: {args.dir}")

            processed: list[dict[str, Any]] = []
            for image_path in images:
                result = process_one_image(
                    processor=processor,
                    model=model,
                    image_path=image_path,
                    prompt=args.prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                saved = save_to_rating_folder(args.dir, image_path, result)
                processed.append(
                    {
                        "source": image_path,
                        "rating": result["rating"],
                        "output_image": saved["image"],
                        "output_json": saved["json"],
                    }
                )
            print(json.dumps({"processed_count": len(processed), "results": processed}, ensure_ascii=True, indent=2))
    except (OSError, RuntimeError) as exc:
        print(f"Model/runtime error: {exc}", file=sys.stderr)
        return 2
    except (KeyError, json.JSONDecodeError, ValueError) as exc:
        print(f"Failed to parse structured model output: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
