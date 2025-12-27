#!/usr/bin/env python3
import os, sys, csv, glob
import numpy as np
import onnxruntime as ort

def parse_args():
    if len(sys.argv) != 4:
        raise SystemExit('Usage: python submission.py <model_path> <images_dir> <output_csv>')
    return sys.argv[1], sys.argv[2], sys.argv[3]

def load_model(model_path):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

def preprocess_image(arr):
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.ndim == 3 and arr.shape[-1] == 6:
        arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - 0.5) / 0.5
    return arr[np.newaxis, :, :, :]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess_data(images_dir):
    arrays_dir = os.path.join(images_dir, "arrays")
    if not os.path.isdir(arrays_dir):
        arrays_dir = images_dir
    for path in sorted(glob.glob(os.path.join(arrays_dir, "*.npy"))):
        file_prefix = os.path.basename(path).replace(".npy", "")
        yield file_prefix, preprocess_image(np.load(path))

def run_inference(session, input_array):
    result = session.run(None, {session.get_inputs()[0].name: input_array})[0]
    return float(softmax(result[0])[1])

def write_csv(predictions, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_prefix', 'error'])
        for file_prefix, prob in predictions:
            writer.writerow([file_prefix, prob])

def main():
    model_path, images_dir, output_csv = parse_args()
    model = load_model(model_path)
    predictions = [(fp, run_inference(model, tensor)) for fp, tensor in preprocess_data(images_dir)]
    write_csv(predictions, output_csv)

if __name__ == '__main__':
    main()
