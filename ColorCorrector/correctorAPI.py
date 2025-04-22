import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pillow_lut import load_cube_file
from flask import Flask, request, jsonify, send_file
from bytesbufio import BytesBufferIO as BytesIO

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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pixels = image.reshape((-1, 3)).astype(float)
    
    mean_vector = np.mean(pixels, axis=0)

    std_vector = np.std(pixels, axis=0)

    standardized_pixels = (pixels - mean_vector) / std_vector

    covariance_matrix = np.cov(standardized_pixels, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    principal_components = eigenvectors[:, :6]

    projected_pixels = np.dot(standardized_pixels, principal_components)

    scaling_factors = generate_scaling_factors(image)

    corrected_pixels = projected_pixels * scaling_factors

    corrected_pixels = np.dot(corrected_pixels, principal_components.T)

    corrected_pixels = corrected_pixels * std_vector + mean_vector

    corrected_image = np.array(corrected_pixels.reshape(image.shape))

    return corrected_image

def apply_lut(image_path, lut_path):
    lut = load_cube_file(lut_path)
    im = Image.open(image_path)
    
    corrected_image_pil = im.filter(lut)
    corrected_image_np = np.array(corrected_image_pil)
    
    return corrected_image_np
    
@app.route('/correctAPI', methods=['POST'])
def color_correct_image():
    if 'file' not in request.files or 'lut_name' not in request.form:
        return jsonify({'error': 'Missing file or LUT name'}), 400

    file = request.files['file']
    lut_name = request.form['lut_name']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            image_np = apply_lut(file, LUT_PATHS.get(lut_name))
            if image_np is None:
                return jsonify({"error": "Failed to apply LUT."}), 500

            corrected_image = color_correction(image_np)
            _, img_encoded = cv2.imencode('.png', corrected_image)

            img_io = BytesIO(img_encoded.tobytes())
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/png')

        except Exception as e:
            return jsonify({'error': f'Error processing image: {e}'}), 500

    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)