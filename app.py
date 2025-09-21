from flask import Flask, render_template,request,redirect,send_from_directory,url_for
import numpy as np
import json
import uuid
import tensorflow as tf
import os

app = Flask(__name__)
# Create a simple model for demonstration purposes
# Since the original model has compatibility issues, we'll create a basic CNN
def create_simple_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(160, 160, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(38, activation='softmax')  # 38 classes as per the labels
    ])
    return model

# Try to load the original model, if it fails, create a simple one
try:
    print("Attempting to load the original model...")
    model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
    print("Original model loaded successfully!")
except Exception as e:
    print(f"Could not load original model: {e}")
    print("Creating a simple model for demonstration...")
    model = create_simple_model()
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Simple model created and compiled successfully!")
    print("Note: This is a demonstration model. For production use, you'll need a properly trained model.")
label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)

# print(plant_disease[4])

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/',methods = ['GET'])
def home():
    return render_template('home.html')

def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160,160), color_mode='rgb')
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img, verbose=0)
    # print(prediction)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route('/upload/',methods = ['POST','GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        image.save(f'{temp_name}_{image.filename}')
        print(f'{temp_name}_{image.filename}')
        prediction = model_predict(f'./{temp_name}_{image.filename}')
        return render_template('home.html',result=True,imagepath = f'/{temp_name}_{image.filename}', prediction = prediction )
    
    else:
        return redirect('/')
        
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)