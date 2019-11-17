# Detoxify

Our project consists of two large parts. The first is the API, which includes the backend of our project and ML models for detecting toxic behavior. The second is a browser extension for Twitter that allows you to hide toxic tweets.

## Detoxify API service

How to start:

`python main.py --nsfw-model nsfw_mobilenet2.224x224.h5 --text-model text_classifier/ --text-replacements text_replacements.txt --image-replacements image_replacements/`

```nsfw_mobilenet2.224x224.h5``` is from https://github.com/GantMan/nsfw_model

## Detoxify browser extension

https://github.com/olpyhn/detoxify-extension 
