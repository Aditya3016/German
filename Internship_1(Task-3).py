# **Task-3**

!pip install pdf2image

!pip install pytesseract

!pip install layoutparser

!apt-get install -y poppler-utils

!pip install torch torchvision torchaudio
!pip install git+https://github.com/facebookresearch/detectron2.git

*DocStruct-Net: Document Structure Understanding Framework*

# Required libraries
import os
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import layoutparser as lp
from layoutparser.models.detectron2 import Detectron2LayoutModel
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Step 1: Convert PDF to Images
images = convert_from_path('sample.pdf', dpi=300)

import layoutparser as lp
print(dir(lp))  # Look for 'Detectron2LayoutModel' in the output

!mkdir -p /root/.torch/iopath_cache/s/dgy9c10wykk4lq4/
!wget -O /root/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth https://www.dropbox.com/s/dgy9c10wykk4lq4/model_final.pth?dl=1

# Step 2: Detect Layout Components
model = lp.models.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    model_path="/root/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth",
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.85],
)
layout = model.detect(images[0])

# Step 3: Crop and Store Components
components = []
for block in layout:
    x1, y1, x2, y2 = map(int, (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))
    cropped_img = images[0].crop((x1, y1, x2, y2))
    text = pytesseract.image_to_string(cropped_img)
    components.append({
        'image': cropped_img,
        'type': block.type,
        'coordinates': (x1, y1, x2, y2),
        'text': text[:200]  # preview
    })

# Step 4: Visual Feature Extraction (CNN)
cnn_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling='avg')
def get_visual_embedding(img):
    img = img.resize((224, 224)).convert('RGB')
    arr = preprocess_input(np.expand_dims(np.array(img), axis=0))
    return cnn_base.predict(arr).flatten()

# Step 5: Textual Feature Extraction (BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=128)
    outputs = bert_model(inputs)[1]  # pooled output
    return outputs.numpy().flatten()

# Step 6: Multimodal Embedding
X = []
y = []
label_map = {'Text': 0, 'Title': 1, 'List': 2, 'Table': 3, 'Figure': 4}
for comp in components:
    vis_emb = get_visual_embedding(comp['image'])
    txt_emb = get_text_embedding(comp['text'])
    combined = np.concatenate((vis_emb, txt_emb))
    X.append(combined)
    y.append(label_map[comp['type']])

# Step 7: Simple Classification Model
input_layer = Input(shape=(len(X[0]),))
x = Dense(256, activation='relu')(input_layer)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(5, activation='softmax')(x)
model = tf.keras.Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(X), np.array(y), epochs=10, batch_size=8)

# Step 8: Prediction and Summary Table
results = []
inv_map = {v: k for k, v in label_map.items()}
for comp in components:
    vis_emb = get_visual_embedding(comp['image'])
    txt_emb = get_text_embedding(comp['text'])
    combined = np.concatenate((vis_emb, txt_emb))
    pred = model.predict(np.expand_dims(combined, axis=0))
    label = inv_map[np.argmax(pred)]
    results.append({
        "Component Type": label,
        "Coordinates": comp['coordinates'],
        "Extracted Text Snippet": comp['text'][:50],
        "Description": f"{label} component at {comp['coordinates']}"
    })

summary_df = pd.DataFrame(results)
summary_df.to_csv("document_structure_summary.csv", index=False)

*Generate Predictions*

predictions = []
for comp in components:
    vis_emb = get_visual_embedding(comp['image'])
    txt_emb = get_text_embedding(comp['text'])
    combined_emb = np.concatenate((vis_emb, txt_emb))

    pred = model.predict(np.expand_dims(combined_emb, axis=0))
    label = np.argmax(pred)
    predictions.append(label)

*Format the Summary as a Table*

inv_label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

import pandas as pd

summary_data = []
for comp, label in zip(components, predictions):
    summary_data.append({
        "Component Type": inv_label_map[label],
        "Coordinates": comp['coordinates'],
        "Text Snippet": comp['text'][:50].strip(),
        "Description": f"{inv_label_map[label]} block at {comp['coordinates']}"
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("DocStruct_Summary.csv", index=False)

# Transformer Encoder Block

from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Layer, Add
from tensorflow.keras.layers import Dense

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(embed_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.add = Add()

    def call(self, x):
        attn_out = self.attn(x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

# Reshape to (batch_size=1, seq_len=num_components, embed_dim)
X_tensor = tf.expand_dims(X_tensor, axis=0)  # Shape: (1, num_components, embed_dim)

transformer_block = TransformerEncoderBlock(embed_dim=X_tensor.shape[-1], num_heads=4)
encoded = transformer_block(X_tensor)
encoded_np = encoded.numpy().squeeze()

# Assume X contains embeddings of all components
# X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
# transformer_block = TransformerEncoderBlock(embed_dim=X_tensor.shape[1], num_heads=4)
# encoded = transformer_block(X_tensor)

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(embed_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.add = Add()

    def build(self, input_shape):
        # Called automatically on first use; no custom weights needed here
        pass

    def call(self, x):
        attn_out = self.attn(x, x)
        x = self.norm1(self.add([x, attn_out]))
        ffn_out = self.ffn(x)
        return self.norm2(self.add([x, ffn_out]))

# Description Generator (Template-Based)

def generate_description(text, comp_type):
    snippet = text.strip().replace("\n", " ")[:80]

    if comp_type == "Title":
        return f"Document title: '{snippet}'"
    elif comp_type == "Text":
        return f"Paragraph content: '{snippet}'"
    elif comp_type == "Table":
        return f"Table with data like: '{snippet}'"
    elif comp_type == "Figure":
        return "Figure or diagram included"
    elif comp_type == "List":
        return f"List includes: '{snippet}'"
    else:
        return f"{comp_type} component with snippet: '{snippet}'"

summary_data = []
for comp in components:
    vis_emb = get_visual_embedding(comp['image'])
    txt_emb = get_text_embedding(comp['text'])
    combined = np.concatenate((vis_emb, txt_emb))

    pred = model.predict(np.expand_dims(combined, axis=0))
    label_idx = np.argmax(pred)
    comp_type = inv_map[label_idx]

    desc = generate_description(comp['text'], comp_type)

    summary_data.append({
        "Component Type": comp_type,
        "Coordinates": comp['coordinates'],
        "Text Snippet": comp['text'][:50].strip(),
        "Description": desc
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("DocStruct_enriched_summary.csv", index=False)

# Multi-Task Model Architecture

*Build Parallel Output Layers*

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

input_layer = Input(shape=(len(X[0]),), name='multimodal_input')

# Shared base
x = Dense(256, activation='relu')(input_layer)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)

# Output 1: Component Type
type_output = Dense(5, activation='softmax', name='type_output')(x)

# Output 2: Description Tag (Dummy Regression or Embedding for now)
# You can replace this with an LSTM decoder if generating full text
desc_output = Dense(128, activation='linear', name='desc_output')(x)

*Compile Multi-Output Model*

model = Model(inputs=input_layer, outputs=[type_output, desc_output])

model.compile(
    optimizer='adam',
    loss={'type_output': 'sparse_categorical_crossentropy', 'desc_output': 'mse'},
    loss_weights={'type_output': 1.0, 'desc_output': 0.5},
    metrics={'type_output': 'accuracy'}
)

*Dummy Training Setup (Extend Later)*

desc_dummy = np.random.rand(len(X), 128)  # Later: use proper text embeddings

model.fit(
    np.array(X),
    {'type_output': np.array(y), 'desc_output': desc_dummy},
    epochs=10,
    batch_size=8
)

# Multi-Page PDF Support

from pdf2image import convert_from_path
import pytesseract
import layoutparser as lp
import numpy as np
import pandas as pd

from google.colab import files
uploaded = files.upload()

!ls /content

# 🔧 Pre-trained layout detection model (local checkpoint path)
layout_model = lp.models.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    model_path="/content/model_final.pth",  # Make sure this file exists
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.85],
)

# 📄 Convert full PDF to image list
images = convert_from_path("sample.pdf", dpi=300)

# 🔍 Collect data across all pages
all_components = []
label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

for page_num, image in enumerate(images):
    layout = layout_model.detect(image)

    for block in layout:
        x1, y1, x2, y2 = map(int, [block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2])
        cropped = image.crop((x1, y1, x2, y2))
        text = pytesseract.image_to_string(cropped)

        all_components.append({
                               "Page": page_num + 1,
                               "Component Type": block.type,
                               "Coordinates": (x1, y1, x2, y2),
                               "Text Snippet": text[:50].strip(),
                               "Description": f"{block.type} component on page {page_num + 1} at {x1,y1,x2,y2}",
                               "image": cropped,         # ✅ Add this line
                               "text": text              # ✅ And this one for full access
                          })

# 📊 Convert to DataFrame and export
df = pd.DataFrame(all_components)
df.to_csv("DocStruct_multi_page_summary.csv", index=False)

# Enriched Multi-Page Summary Generator

import numpy as np
import pandas as pd
import tensorflow as tf

# Assumes the following are already initialized elsewhere:
# - all_components (list of parsed document regions with image + text)
# - transformer_block (e.g. TransformerEncoderBlock(embed_dim, num_heads))
# - model (multi-task Keras model)
# - get_visual_embedding(), get_text_embedding() functions
# - inv_map (index-to-label mapping)

# 🔎 Step 1: Generate multimodal embeddings
X = []
for comp in all_components:
    vis_emb = get_visual_embedding(comp['image'])
    txt_emb = get_text_embedding(comp['text'])
    combined = np.concatenate((vis_emb, txt_emb))
    X.append(combined)

# 🧠 Step 2: Encode embeddings with transformer
X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
X_tensor = tf.expand_dims(X_tensor, axis=0)  # shape: (1, num_components, embed_dim)
encoded = transformer_block(X_tensor)        # output: enriched embeddings
encoded_np = encoded.numpy().squeeze()

# 🎯 Step 3: Predict type and generate descriptions
results = []
for idx, comp in enumerate(all_components):
    pred = model.predict(np.expand_dims(encoded_np[idx], axis=0))
    label_idx = np.argmax(pred[0])  # type classifier output
    comp_type = inv_map[label_idx]

    text = comp['text'].strip().replace("\n", " ")
    snippet = text[:80]

    # ✏️ Description logic
    if comp_type == "Title":
        desc = f"Document title: '{snippet}'"
    elif comp_type == "Text":
        desc = f"Paragraph content: '{snippet}'"
    elif comp_type == "Table":
        desc = f"Table with data like: '{snippet}'"
    elif comp_type == "Figure":
        desc = "Figure or diagram included"
    elif comp_type == "List":
        desc = f"List includes: '{snippet}'"
    else:
        desc = f"{comp_type} component: '{snippet}'"

    results.append({
        "Page": comp['Page'],
        "Component Type": comp_type,
        "Coordinates": comp['Coordinates'],
        "Text Snippet": comp['text'][:50].strip(),
        "Description": desc
    })

# 📊 Step 4: Export to summary table
df = pd.DataFrame(results)
df.to_csv("DocStruct_enriched_multi_page_summary.csv", index=False)

*Model Setup & Utilities*

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.applications import MobileNetV2
from transformers import BertTokenizer, TFBertModel

# 🔧 Load CNN and BERT models
cnn_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling='avg')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# 🧠 Define embedding extractors
def get_visual_embedding(img):
    img = img.resize((224, 224)).convert('RGB')
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(np.array(img), axis=0))
    return cnn_base.predict(arr).flatten()

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=128)
    outputs = bert_model(inputs)[1]  # pooled output
    return outputs.numpy().flatten()

*Region Collection (Multi-Page Parsing)*

# Example loop inside your full-page parsing block
all_components.append({
    "Page": page_num + 1,
    "Component Type": block.type,
    "Coordinates": (x1, y1, x2, y2),
    "Text Snippet": text[:50].strip(),
    "Description": f"{block.type} component on page {page_num + 1} at {x1,y1,x2,y2}",
    "image": cropped,
    "text": text
})

*Model Persistence*

# Save the model after training
model.save("docstruct_multitask_model.keras")

# Reload it later safely
from tensorflow.keras.models import load_model
model = load_model("docstruct_multitask_model.h5", custom_objects={'mse': MeanSquaredError()})
