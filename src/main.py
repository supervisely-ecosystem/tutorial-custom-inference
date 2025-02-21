import os

import supervisely as sly
from dotenv import load_dotenv

from src.custom_model import CustomModel

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


api = sly.Api.from_env()
project_id = sly.env.project_id()

m = CustomModel(use_gui=True, use_serving_gui_template=True)
m.serve()

# # ⬇ This code below is for running as a simple script (not as a serving app) ⬇

# # Load model
# deploy_params = {"device": "cpu"}
# m._load_model(deploy_params)

# # Inference
# test_images = sly.fs.list_files_recursively("src/demo_data/input", valid_extensions=[".jpg"])
# inf_settings = {
#     "conf": 0.25,
#     "iou": 0.7,
#     "half": False,
#     "max_det": 300,
#     "agnostic_nms": False,
#     "return_heatmaps": True,
# }
# anns = m.inference(test_images, inf_settings)

# # Visualize results
# from src.render_heatmaps import render_heatmaps_on_image

# output_dir = "src/demo_data/output"
# sly.fs.mkdir(output_dir, remove_content_if_exists=True)
# for img_path, ann in zip(test_images, anns):
#     img = sly.image.read(img_path)
#     # ann.draw(img)
#     # or
#     img = render_heatmaps_on_image(img_path, ann)
#     sly.image.write(os.path.join(output_dir, os.path.basename(img_path)), img)

# print("✅ Success!")
