import os

import supervisely as sly
from dotenv import load_dotenv

from render_heatmaps import render_heatmaps_on_image

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()
image_path = "src/demo_data/input/2.jpg"  # ⬅︎ Put your image path here
app_url = "http://localhost:8000"

session = sly.nn.inference.Session(
    api, session_url=app_url, inference_settings={"return_heatmaps": True}
)

ann = session.inference_image_path(image_path)
# print(ann)

output_dir = "src/demo_data/output"
sly.fs.mkdir(output_dir, remove_content_if_exists=True)
img = sly.image.read(image_path)
# ann.draw(img) # ⬅︎ draw predictions on the image
# or render heatmaps
img = render_heatmaps_on_image(image_path, ann)
sly.image.write(os.path.join(output_dir, os.path.basename(image_path)), img)

print("✅ Success!")
