
import os
import gradio as gr
from fastai.vision.all import *

learn = load_learner("export.pkl")
labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, proba = learn.predict(img)
    return {
        labels[i]: float(proba[i])
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Bear Classifier"
)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
