import os

import supervisely as sly
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()
image_path = "src/demo_data/input/2.jpg"  # ⬅︎ Put your image path here
app_url = "http://localhost:8000"

session = sly.nn.inference.SessionJSON(
    api, session_url=app_url, inference_settings={"return_heatmaps": True}
)

prediction = session.inference_image_path(image_path)
print(prediction)
print("✅ Success!")
