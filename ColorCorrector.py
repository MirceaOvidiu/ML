import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def generate_scaling_factors(image):
    # Imaginea este transformata intr-un vector bidimensional de pixeli.
    pixels = image.reshape((-1, 3)).astype(float)

    mean_vector = np.mean(pixels, axis=0) # Media pe fiecare scope - R, G, B
    std_vector = np.std(pixels, axis=0) # deviatia standard pe fiecare coloana
    standardized_pixels = (pixels - mean_vector) / std_vector # ne asiguram ca fiecare canal are aceeasi scala, standardizand datele

    # Cream un obiect de tip PCA, cu 3 componente principale
    pca = PCA(n_components=3)
    # Se aplica PCA pe datele standardizate
    # si trecem in alt sistem de coordonate, unde axele sunt componentele principale
    principal_components = pca.fit_transform(standardized_pixels)

    # Calcularea factorilor de scalare
    # Impartind dispersia la media valorilor absolute ale componentelor principale
    # "aplatizam" variatiile si aceasta metoda ofera culori cat mai naturale 
    # si de asemenea s-a dovedit a fi robusta la highlight-uri, umbre si undertone-uri
    scaling_factors = np.std(principal_components, axis=0) / np.mean(np.abs(principal_components), axis=0)

    return scaling_factors

def color_correction(image):
    # Imaginea este transformata intr-un vector bidimensional.
    pixels = image.reshape((-1, 3)).astype(float)
    
    # Se calculeaza valoarea medie a pixelilor din fiecare scope - R, G, B
    mean_vector = np.mean(pixels, axis=0)

    # Se calculeaza deviatia standard a pixelilor din fiecare scope - R, G, B
    std_vector = np.std(pixels, axis=0)

    # Se standardizeaza pixelii, penutr a scala datele
    standardized_pixels = (pixels - mean_vector) / std_vector

    # Cream un obiect de tip PCA cu 3 componente principale
    pca = PCA(n_components=3)


    principal_components = pca.fit_transform(standardized_pixels)

    # Generam factorii de scalare, conform functiei precedente
    scaling_factors = generate_scaling_factors(image)

    # Ajustam componentele principale pentru corectie
    corrected_components = principal_components * scaling_factors
    # Astfel, ajustam variatia fiecarei componente principale
    # Si corectam imbalansurile de culoare


    # Corectam pixelii
    corrected_pixels = np.dot(corrected_components, pca.components_)
    # Trecem inapoi la sistemul de coordonate initial - R, G, B
    
    # De-standardizam pixelii
    corrected_pixels = corrected_pixels * std_vector + mean_vector

    # Revenim la forma initiala a imaginii
    corrected_image = corrected_pixels.reshape(image.shape)

    # Ne asiguram suntem in spatiul 8 bit de culoare.
    return np.clip(corrected_image, 0, 255).astype(np.uint8)


def plot_difference_heatmap(original_image, corrected_image):
    # Calculam diferenta dintre imaginea originala si cea corectata
    diff_image = np.abs(original_image - corrected_image)

    # Calculam norma 2 a diferentelor pe fiecare pixel
    RGB_scope_difference = np.linalg.norm(diff_image, axis=2)

    # Si afisam rezultatul sub forma de heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(RGB_scope_difference, cmap='coolwarm')
    plt.title('Pixel-wise Difference Heatmap')
    plt.show()

print("Color Corrector")
print("===================================")

print("Introduceti path-ul imaginii:")
image_path = input()

# Extragem numele imaginii
image_name = image_path.split("/")[-1].split(".")[0]

original_image = cv2.imread(image_path)

corrected_image = color_correction(original_image)

original_image = cv2.resize(original_image, None, fx=1, fy=1)

cv2.imshow("Original Image", original_image)

corrected_image = cv2.resize(corrected_image, None, fx=1, fy=1)

cv2.imshow("Color Corrected Image", corrected_image)
cv2.imwrite("corrected_image.jpg", corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

hist_original = cv2.calcHist([original_image], [0], None, [256], [0, 256])
hist_corrected = cv2.calcHist([corrected_image], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 10))
plt.title("Histograms")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist_original, label="Original")
plt.plot(hist_corrected, label="Corrected")

plot_difference_heatmap(original_image, corrected_image)