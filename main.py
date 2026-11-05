from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DIR = r'D:\политех\4sem\архитектура\cursovik\archive\skin burn dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'valid')


IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 20

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
  TRAIN_DIR,
  target_size=IMG_SIZE,
  batch_size=BATCH_SIZE,
  class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
  VAL_DIR,
  target_size=IMG_SIZE,
  batch_size=BATCH_SIZE,
  class_mode='categorical'
)


model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
  MaxPooling2D((2, 2)),
    
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
    
  Conv2D(128, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
    
  Flatten(),
  Dense(256, activation='relu'),
  Dropout(0.5),
  Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
  EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
  ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(
  train_generator,
  epochs=EPOCHS,
  validation_data=val_generator,
  callbacks=callbacks,
  verbose=1
)

val_loss, val_acc = model.evaluate(val_generator)
print(f"\nТочность: {val_acc:.4f} ({val_acc*100:.2f}%)")

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Точность модели')
plt.savefig('accuracy_graph.png')
plt.show()