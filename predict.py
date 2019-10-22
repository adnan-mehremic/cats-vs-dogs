from PIL import Image
import cv2
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

CATEGORIES = ["Cat", "Dog"]

def prepare(filepath):
    IMG_SIZE = 150
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

im = Image.open('/path/to/image')
im.show()

prediction = model.predict([prepare('/path/to/image')])
print(prediction)
print(CATEGORIES[int(prediction[0][0])])
