import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

MODEL_PATH = r'D:\политех\4sem\архитектура\cursovik\models\best_burn_model.keras'

if not os.path.exists(MODEL_PATH):
  print(f"Ошибка: файл модели не найден по пути: {MODEL_PATH}")
  exit()

model = tf.keras.models.load_model(MODEL_PATH)

def predict_burn(image_path):
  if not os.path.exists(image_path):
    print(f"Файл {image_path} не найден")
    return None, 0

  img = image.load_img(image_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0) / 255.0

  pred = model.predict(img_array, verbose=0)[0]
  classes = ['1 степень (поверхностный)', '2 степень (глубокий дермальный)', '3 степень (полная толщина)']
    
  result = classes[np.argmax(pred)]
  confidence = np.max(pred) * 100
    
  print("="*50)
  print(f"СТЕПЕНЬ ОЖОГА: {result}")
  print(f"Уверенность: {confidence:.1f}%")
  print("="*50)
    
  print("\nВсе вероятности:")
  for i, prob in enumerate(pred):
    print(f"   {classes[i]}: {prob*100:.1f}%")
    
  return result, confidence

PHOTO_PATH = r'D:\политех\4sem\архитектура\cursovik\test\1_1.jpg'

predict_burn(PHOTO_PATH)