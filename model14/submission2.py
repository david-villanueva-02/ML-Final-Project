#!/usr/bin/env python3

import os
import sys
import csv
import glob
import numpy as np
import onnxruntime as ort

def parse_args():
    """
    Exactly three positional arguments:
      1) model_path: path to the ONNX model file
      2) images_dir: directory with test images (same structure as training data)
      3) output_csv: path where CSV with predictions must be stored
    """
    if len(sys.argv) != 4:
        raise SystemExit('Usage: python submission.py <model_path> <images_dir> <output_csv>')

    model_path = sys.argv[1]
    images_dir = sys.argv[2]
    output_csv = sys.argv[3]

    if not os.path.isfile(model_path):
        raise FileNotFoundError('Model file not found: ' + model_path)
    
    if not os.path.isdir(images_dir):
        raise NotADirectoryError('Images directory not found: ' + images_dir)

    return model_path, images_dir, output_csv


def load_model(model_path):
    """
    TO DO:
    Load ONNX model and return an inference session (or equivalent).
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session


def preprocess_image(arr):
    '''
    Preprocess a single image array as required by the model.
    '''
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.ndim == 3 and arr.shape[-1] == 6:
        arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - 0.5) / 0.5
    arr = arr[np.newaxis, :, :, :]
    return arr


def softmax(x):
    '''
    Compute softmax values for each sets of scores in x.
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess_data(images_dir):
    """
    TO DO:
    Load and preprocess data as required by the model.
    Return an iterator over input tensors and file_prefix.
    """
    arrays_dir = os.path.join(images_dir, "arrays")
    if not os.path.isdir(arrays_dir):
        arrays_dir = images_dir

    npy_files = sorted(glob.glob(os.path.join(arrays_dir, "*.npy")))

    if len(npy_files) == 0:
        raise RuntimeError(f"No .npy files found in {arrays_dir}")

    for path in npy_files:
        file_prefix = os.path.basename(path).replace(".npy", "")
        arr = np.load(path)
        input_tensor = preprocess_image(arr)
        yield file_prefix, input_tensor


def run_inference(session, input_array):
    """
    TO DO:
    Run the ONNX model on a single preprocessed input and return predictions.
    Convert raw model output (logit) to post-softmax confidence score.
    store the results in this format: [file_prefix, probability]
    """
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_array})[0]
    logits = result[0]
    probs = softmax(logits)
    return float(probs[1])


def write_csv(predictions, output_csv):
    """
    Write predictions to CSV.
    Expected format:
      file_prefix,error
    """
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_prefix', 'error'])
        for file_prefix, error_conf in predictions:
            writer.writerow([file_prefix, error_conf])


def main():
    model_path, images_dir, output_csv = parse_args()
    model = load_model(model_path)

    predictions = []
    for file_prefix, input_tensor in preprocess_data(images_dir):
        error_conf = run_inference(model, input_tensor)
        predictions.append((file_prefix, error_conf))

    write_csv(predictions, output_csv)


if __name__ == '__main__':
    main()
