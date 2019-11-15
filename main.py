#!/usr/bin/env python3

import argparse
import os

import api
import worker


def main():
    args = parse_args()

    worker.load_assets(args.nsfw_model, args.text_model)

    with open(args.text_replacements, "r") as text_replacements_file:
        text_replacements = \
            parse_text_replacements(text_replacements_file.read())

    image_replacements = load_image_replacements(args.image_replacements)

    api.DetoxifyApi.run_app(
        args.max_workers,
        text_replacements,
        image_replacements,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detoxify API server",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        help="max number of workers",
    )

    parser.add_argument(
        "--nsfw-model",
        required=True,
        help="path to NSFW classifier model",
    )

    parser.add_argiment(
        "--text-model",
        required=True,
        help="path to text classifier models",
    )

    parser.add_argument(
        "--text-replacements",
        required=True,
        help="path to a ###-separated list of toxic text replacements",
    )

    parser.add_argument(
        "--image-replacements",
        required=True,
        help="path to toxic image replacement directory",
    )

    return parser.parse_args()


def parse_text_replacements(s):
    return [x.strip() for x in s.split("###")]


def load_image_replacements(path):
    image_replacements = []

    for name in os.listdir(path):
        with open(f"{path}/{name}", "rb") as file:
            image_replacements.append(file.read())

    return image_replacements


if __name__ == "__main__":
    main()
