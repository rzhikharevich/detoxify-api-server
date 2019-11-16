import aiohttp.web as aioweb
import asyncio
import base64
import concurrent.futures
import contextlib
import io

from PIL import Image

import utils
import worker


class DetoxifyApi:
    def __init__(self, executor, text_replacements, image_replacements):
        self.executor = executor
        self.text_replacements = text_replacements
        self.image_replacements = [*map(make_data_url, image_replacements)]

    @classmethod
    def run_app(cls, max_workers, text_replacements, image_replacements):
        with \
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=worker.init,
                ) as executor:
            api = cls(executor, text_replacements, image_replacements)

            app = aioweb.Application()
            app.add_routes([
                aioweb.get("/", api.api_index),
                aioweb.post("/detoxify", api.api_detoxify),
            ])

            async def create_app():
                await api.offload(worker.wait_for_init)
                return app

            aioweb.run_app(create_app())

    async def api_index(self, request):
        return aioweb.Response(text="Welcome to Detoxify API!")

    async def api_detoxify(self, request):
        body = await request.json()

        response = {}

        if \
                "text" in body and \
                await self.offload(worker.is_toxic_text, body["text"]):
            response["text"] = utils.pick_random(self.text_replacements)
            print(response)

        if \
                "image" in body and \
                await self.offload(worker.is_toxic_image, body["image"]):
            response["image"] = utils.pick_random(self.image_replacements)

        return aioweb.json_response(response)

    async def offload(self, fun, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, fun, *args)


def make_data_url(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    fmt = image.format.lower()

    image_base64 = base64.b64encode(image_bytes).decode()

    return f"data:image/{fmt};base64,{image_base64}"
