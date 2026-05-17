from tensorflow.keras.models import load_model

from preprocess import preprocess_image_input


model_path='../model/densenet_freeze.h5'
model= load_model(model_path)


def predict_output(input_image):
    
    preprocessed_image=preprocess_image_input(input_image)
    
    prediction= model.predict(preprocessed_image)
    
    probability_pneumonia= float(prediction[0][0])
    
    if probability_pneumonia>0.5:
        predicted_class='Pneumonia'
        likelihood= "High Likelihood for Pneumonia"
        confidence= probability_pneumonia*100
    elif 0.4<=probability_pneumonia<=0.6:
        predicted_class='Maybe Pneumonia'
        likelihood= "Uncertain Prediction"
        confidence= probability_pneumonia*100
    else:
        predicted_class='Normal'
        likelihood= "Low Likelihood of Pneumonia"
        confidence= (1-probability_pneumonia)*100
    
    confidence = min(confidence, 99.99)
    return {
    "predicted_class": predicted_class,
    "likelihood": likelihood,
    "confidence": round(confidence, 2)
}