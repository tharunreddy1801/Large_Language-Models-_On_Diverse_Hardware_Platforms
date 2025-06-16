import torch
import onnx
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
input_sentence = "I love this product!"
input_ids = tokenizer(input_sentence, return_tensors="pt")["input_ids"]

model.to("cpu")  # Ensure the model is on CPU for export
onnx_path = "yourmodel.onnx"  # Specify the output path for the ONNX file
torch.onnx.export(model,(input_ids), onnx_path, opset_version=11)
print("success:",onnx_path)