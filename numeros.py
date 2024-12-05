from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import logging

from urllib import parse
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Carga de datos de MNIST (set de entrenamiento y de pruebas)
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Normalizar: Números de 0 a 255, que sean de 0 a 1
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# Estructura de la red
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # Para clasificación
])

# Configurar el modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
BATCHSIZE = 32
train_dataset = train_dataset.repeat().shuffle(metadata.splits['train'].num_examples).batch(BATCHSIZE)
test_dataset = test_dataset.batch(BATCHSIZE)

model.fit(
    train_dataset, epochs=5,
    steps_per_epoch=math.ceil(metadata.splits['train'].num_examples / BATCHSIZE)
)

# Clase para definir el servidor HTTP
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        # Respuesta básica para solicitudes GET
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Access-Control-Allow-Origin", "*")  # Evitar problemas con CORS
        self.end_headers()
        self.wfile.write(b"Servidor funcionando. Usa POST para enviar datos.")

    def do_POST(self):
        print("Petición recibida")

        # Obtener datos de la petición y limpiar los datos
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        data = data.decode().replace('pixeles=', '')
        data = parse.unquote(data)

        # Transformar datos para que coincidan con el formato MNIST
        arr = np.fromstring(data, np.float32, sep=",")
        arr = arr.reshape(28, 28)
        arr = np.array(arr)
        arr = arr.reshape(1, 28, 28, 1)

        # Realizar y obtener la predicción
        prediction_values = model.predict(arr, batch_size=1)
        prediction = str(np.argmax(prediction_values))
        print("Predicción final: " + prediction)

        # Responder a la petición HTTP
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(prediction.encode())

# Iniciar el servidor en el puerto 8000
print("Iniciando el servidor...")
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
server.serve_forever()
