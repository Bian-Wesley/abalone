import tensorflow as tf
#could also experiment with json
model = tf.keras.models.load_model("onefile.h5")

print(model.predict([[0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.1]]))
