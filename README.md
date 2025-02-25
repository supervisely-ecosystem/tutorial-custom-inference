# Custom Model Inference Integration Tutorial

This is a source code repository for the tutorial on how to integrate a custom model inference class. The tutorial is available at [https://developer.supervisely.com/app-development/neural-network-integration/inference/custom-inference-with-alpha-mask-segmentation](https://developer.supervisely.com/app-development/neural-network-integration/inference/custom-inference-with-alpha-mask-segmentation)

## How to use

1. Clone this repository

```bash
git clone git@github.com:supervisely-ecosystem/tutorial-custom-inference.git
cd tutorial-custom-inference
```

2. Install the required dependencies

We recommend using a virtual environment to install the dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run the example

```bash
uvicorn src.main:m.app --host 0.0.0.0 --port 8000 --ws websockets
```

The app will be available at <a href="http://localhost:8000" target="_blank">http://localhost:8000</a>.

By clicking the `Serve` button, you can deploy the model.

4. Get the prediction

Now, when the model is deployed locally, you can connect to it and make predictions.

Run:

```bash
python src/session_inference.py
```

The script will send the image to the model and print the prediction.

5. Visualize the prediction

You can visualize the prediction by rendering the heatmaps on the image. Add the following code to the `session_inference.py` script:

```python
from render_heatmaps import render_heatmaps_on_image

output_dir = "src/demo_data/output"
sly.fs.mkdir(output_dir, remove_content_if_exists=True)
img = sly.image.read(image_path)
# ann.draw(img) # ⬅︎ draw predictions on the image
# or render heatmaps
img = render_heatmaps_on_image(image_path, ann)
sly.image.write(os.path.join(output_dir, os.path.basename(image_path)), img)
```

and run the script again.
