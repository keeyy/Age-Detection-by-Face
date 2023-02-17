import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model('model sequential improve.h5')

# Custom function to load and predict label for the image
def predict(img_rel_path):
    # Import Image from the path with size of (300, 300)
    img = Image.open(img_rel_path).resize((150, 150))

    # Convert Image to a numpy array
    img = np.array(img)

    # Scaling the Image Array values between 0 and 1
    img = img / 255.0

    # Get the Predicted Label for the loaded Image
    p = model.predict(img[np.newaxis, ...])

    # Label array
    labels = {0: 'baby', 1: 'kid', 2: 'young', 3: 'adult'}

    predicted_class = labels[np.argmax(p[0], axis=-1)]

    classes=[]
    prob=[]
    for i,j in enumerate (p[0],0):
        classes.append(labels[i])
        prob.append(round(j*100,2))

    return predicted_class, classes, prob

def main():
    st.title("Face Detection")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button("Predict"):
            class_, classes, prob = predict(uploaded_file)
            st.write("Age:", class_)
            st.write("Predict:")
            for i in range(len(classes)):
                st.write(f"{classes[i].upper()}: {prob[i]}%")

if __name__ == "__main__":
    main()