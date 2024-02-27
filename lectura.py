from keras.models import load_model
import cv2
import numpy as np

# Cargar el modelo
model = load_model('modelo.h5')

image_path = input("Por favor, ingrese la ruta de la imagen: ")

image_path = image_path.strip('"')

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 64))
img = np.array(img).reshape(-1, 64, 64, 1)

prediction = model.predict(img)

classes = ['verde', 'maduro', 'podrido', 'semi_maduro', 'semi_podrido']
predicted_class = classes[np.argmax(prediction)]

print(f'La clase predicha es: {predicted_class}')