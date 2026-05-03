import json
import numpy as np
import gradio as gr
import onnxruntime as ort
from PIL import Image

with open("metadata.json") as f:
    meta = json.load(f)

CLASS_NAMES    = meta["class_names"]
IMG_SIZE       = meta["img_size"]
MEAN           = np.array(meta["mean"], dtype=np.float32)
STD            = np.array(meta["std"],  dtype=np.float32)
CONF_THRESHOLD = meta["conf_threshold"]
PER_CLASS_F1   = meta["per_class_f1"]

EMOJIS = {
    "butterfly":"🦋","cat":"🐱","chicken":"🐔","cow":"🐄",
    "dog":"🐕","elephant":"🐘","horse":"🐎","sheep":"🐑",
    "spider":"🕷️","squirrel":"🐿️"
}

sess       = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.expand_dims(arr.transpose(2,0,1), 0)

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def classify(img: Image.Image):
    if img is None:
        return build_empty_html()

    logits = sess.run(None, {INPUT_NAME: preprocess(img)})[0][0]
    probs  = softmax(logits)
    conf   = float(probs.max())
    idx    = int(probs.argmax())
    cls    = CLASS_NAMES[idx]
    emoji  = EMOJIS[cls]
    f1     = PER_CLASS_F1[cls]
    uncertain = conf < CONF_THRESHOLD

    top5 = sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1])[:5]

    # circular arc: circumference = 2π×18 ≈ 113
    circ   = 113
    offset = circ * (1 - conf)
    color  = "#E24B4A" if uncertain else "#1D9E75"
    status_color = "#BA7517" if uncertain else "#1D9E75"
    status_text  = f"⚠ Low confidence — may be out of distribution" if uncertain else "✓ High confidence prediction"

    bars_html = ""
    for name, prob in top5:
        pct   = prob * 100
        width = max(pct, 1.5)
        bar_color = "#1D9E75" if name == cls else "#B4B2A9"
        bold  = "font-weight:500; color:var(--color-text-primary);" if name == cls else "color:var(--color-text-secondary);"
        bars_html += f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
            <span style="{bold}">{EMOJIS[name]} {name}</span>
            <span style="color:var(--color-text-secondary); font-size:12px;">{pct:.1f}%</span>
          </div>
          <div style="height:3px; background:var(--color-border-tertiary); border-radius:2px;">
            <div style="width:{width}%; height:100%; background:{bar_color}; border-radius:2px; transition:width 0.4s ease;"></div>
          </div>
        </div>"""

    html = f"""
    <div style="font-family:var(--font-sans); padding:0.5rem 0;">

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:start;">

        <div style="background:var(--color-background-primary); border:0.5px solid var(--color-border-tertiary); border-radius:var(--border-radius-lg); padding:1.25rem;">
          <p style="font-size:11px; font-weight:500; color:var(--color-text-secondary); text-transform:uppercase; letter-spacing:0.06em; margin:0 0 1rem;">Prediction</p>

          <div style="display:flex; align-items:center; gap:14px; padding-bottom:1rem; margin-bottom:1rem; border-bottom:0.5px solid var(--color-border-tertiary);">
            <div style="width:52px; height:52px; border-radius:var(--border-radius-lg); background:var(--color-background-secondary); display:flex; align-items:center; justify-content:center; font-size:26px; flex-shrink:0;">{emoji}</div>
            <div style="flex:1; min-width:0;">
              <p style="font-size:22px; font-weight:500; color:var(--color-text-primary); margin:0; text-transform:capitalize;">{cls}</p>
              <p style="font-size:12px; color:{status_color}; margin:2px 0 0; white-space:nowrap;">{status_text}</p>
            </div>
            <div style="position:relative; width:44px; height:44px; flex-shrink:0;">
              <svg viewBox="0 0 44 44" width="44" height="44">
                <circle cx="22" cy="22" r="18" fill="none" stroke="var(--color-border-tertiary)" stroke-width="3"/>
                <circle cx="22" cy="22" r="18" fill="none" stroke="{color}" stroke-width="3"
                  stroke-dasharray="{circ}" stroke-dashoffset="{offset:.1f}"
                  stroke-linecap="round" transform="rotate(-90 22 22)"/>
              </svg>
              <span style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:10px; font-weight:500; color:var(--color-text-primary);">{conf*100:.0f}%</span>
            </div>
          </div>

          <div style="background:var(--color-background-secondary); border-radius:var(--border-radius-md); padding:10px 12px;">
              <p style="font-size:11px; color:var(--color-text-secondary); margin:0 0 2px;">Confidence score</p>
              <p style="font-size:20px; font-weight:500; color:var(--color-text-primary); margin:0;">{conf*100:.1f}%</p>
          </div>
        </div>

        <div style="background:var(--color-background-primary); border:0.5px solid var(--color-border-tertiary); border-radius:var(--border-radius-lg); padding:1.25rem;">
          <p style="font-size:11px; font-weight:500; color:var(--color-text-secondary); text-transform:uppercase; letter-spacing:0.06em; margin:0 0 1rem;">Top 5 predictions</p>
          {bars_html}
        </div>

      </div>

      <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:1rem; align-items:center;">
        <span style="font-size:12px; color:var(--color-text-secondary); margin-right:2px;">Supports:</span>
        {"".join(f'<span style="font-size:12px; padding:3px 10px; border-radius:20px; background:var(--color-background-secondary); border:0.5px solid var(--color-border-tertiary); color:var(--color-text-secondary);">{EMOJIS[c]} {c}</span>' for c in sorted(CLASS_NAMES))}
      </div>

    </div>"""
    return html

def build_empty_html():
    return """
    <div style="font-family:var(--font-sans); padding:2rem; text-align:center; color:var(--color-text-secondary);">
      <div style="font-size:40px; margin-bottom:12px;">🤔</div>
      <p style="font-size:14px;">Upload an animal photo to see the prediction</p>
    </div>"""

CSS = """
.gradio-container { max-width: 860px !important; margin: 0 auto !important; }
footer { display: none !important; }
#upload-box { border-radius: 12px !important; }
.gr-prose h1 { font-size: 22px !important; font-weight: 500 !important; }
button.primary { background: #534AB7 !important; border-color: #534AB7 !important; color: white !important; border-radius: 8px !important; }
button.primary:hover { background: #3C3489 !important; border-color: #3C3489 !important; }
"""

with gr.Blocks(title="Animal Classifier") as demo:

    gr.HTML("""
    <div style="font-family:var(--font-sans); padding:1.5rem 0 0.5rem;">
      <div style="display:flex; align-items:baseline; gap:10px; margin-bottom:4px;">
        <span style="font-size:22px; font-weight:500; color:var(--color-text-primary);">Animal classifier</span>
        <span style="font-size:12px; padding:2px 10px; border-radius:20px; background:var(--color-background-secondary); border:0.5px solid var(--color-border-tertiary); color:var(--color-text-secondary);">EfficientNet-B3 · 98.80% acc · ONNX</span>
      </div>
      <p style="font-size:14px; color:var(--color-text-secondary); margin:0;">Upload any animal photo for instant identification across 10 classes</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                type="pil",
                label="Upload image",
                elem_id="upload-box",
                height=280,
                show_label=False,
            )
            btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=2):
            result_html = gr.HTML(value=build_empty_html())

    gr.Examples(
        examples=[
            ["examples/horse.jpeg"],
            ["examples/horse-2.jpeg"],
            ["examples/spider.jpeg"],
        ],
        inputs=img_input,
        label="Try an example",
    )

    btn.click(fn=classify, inputs=img_input, outputs=result_html)

demo.launch(css=CSS)
