import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pillow_lut import load_cube_file
from flask import Flask, request, jsonify

app = Flask(__name__)

LUT_PATHS = {
    r"LUTs\Canon C-Log2 to Rec.709 LUT 33x33.cube": r"LUTs\Canon C-Log2 to Rec.709 LUT 33x33.cube",
    r"LUTs\Canon C-Log3 to Rec.709 LUT 33x33.cube": r"LUTs\Canon C-Log3 to Rec.709 LUT 33x33.cube",
    r"LUTs\DJI D-Log to Rec.709 LUT 33x33.cube": r"LUTs\DJI D-Log to Rec.709 LUT 33x33.cube",
    r"LUTs\Fujifilm F-Log to Rec.709 LUT 33x33.cube": r"LUTs\Fujifilm F-Log to Rec.709 LUT 33x33.cube",
    r"LUTs\Nikon N-Log to Rec.709 LUT 33x33.cube": r"LUTs\Nikon N-Log to Rec.709 LUT 33x33.cube",
    r"LUTs\Sony S-Log2 to Rec.709 LUT 33x33.cube": r"LUTs\Sony S-Log2 to Rec.709 LUT 33x33.cube",
    r"LUTs\Sony S-Log3 to Rec.709 LUT 33x33.cube": r"LUTs\Sony S-Log3 to Rec.709 LUT 33x33.cube"
}

def generate_scaling_factors(image):
    pixels = image.reshape((-1, 3)).astype(float)

    mean_vector = np.mean(pixels, axis=0)
    std_vector = np.std(pixels, axis=0)
    standardized_pixels = (pixels - mean_vector) / std_vector

    covariance_matrix = np.cov(standardized_pixels, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    principal_components = eigenvectors[:, :3]

    transformed_pixels = standardized_pixels.dot(principal_components)

    scaling_factors = np.std(transformed_pixels, axis=0) / np.mean(
        np.abs(transformed_pixels), axis=0
    )

    return scaling_factors

def color_correction(image):
    pixels = image.reshape((-1, 3)).astype(float)
    
    mean_vector = np.mean(pixels, axis=0)

    std_vector = np.std(pixels, axis=0)

    standardized_pixels = (pixels - mean_vector) / std_vector

    covariance_matrix = np.cov(standardized_pixels, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    principal_components = eigenvectors[:, :5]

    projected_pixels = np.dot(standardized_pixels, principal_components)

    scaling_factors = generate_scaling_factors(image)

    corrected_pixels = projected_pixels * scaling_factors

    corrected_pixels = np.dot(corrected_pixels, principal_components.T)

    corrected_pixels = corrected_pixels * std_vector + mean_vector

    corrected_image = corrected_pixels.reshape(image.shape)

    return np.clip(corrected_image, 0, 255).astype(np.uint8)
    
def component_histograms(original_image, image_title):
    channels = cv2.split(original_image)

    colors = ("b", "g", "r")
    plt.figure(figsize=(10, 10))
    plt.title(image_title + " separate scopes histograms")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    # For each channel: calculate histogram, plot it, calculate median, plot vertical line
    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        median = np.median(channel)
        plt.axvline(median, color=color, linestyle='dashed', linewidth=2)

    plt.legend(['Blue', 'Green', 'Red', 'Median Blue', 'Median Green', 'Median Red'])
    plt.show()    

def apply_lut(image_path, lut_path):
    lut = load_cube_file(lut_path)
    im = Image.open(image_path)
    
    corrected_image_pil = im.filter(lut)
    corrected_image_np = np.array(corrected_image_pil)
    
    return corrected_image_np
    
@app.route('/color_correct', methods=['POST'])
def color_correct_image():
    if 'file' not in request.files or 'lut_name' not in request.form:
        return jsonify({'error': 'Missing file or LUT name'}), 400

    file = request.files['file']
    lut_name = request.form['lut_name']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image_bytes = file.read()
        lut_path = LUT_PATHS.get(lut_name)
        if lut_path is None:
            return jsonify({"error": "Invalid LUT name provided."}), 400

        lut_applied_image = apply_lut(image_bytes, lut_path)
        if lut_applied_image is None:
            return jsonify({"error": "Failed to apply LUT."}), 500

        corrected_image = color_correction(lut_applied_image)
        _, img_encoded = cv2.imencode('.jpg', corrected_image)
        return jsonify({'image': img_encoded.tobytes().decode('latin-1')}) # important change here.
    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run(debug=True)