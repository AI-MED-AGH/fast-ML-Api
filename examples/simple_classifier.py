"""
Simple classifier example using FastMLAPI.

Run with:
    python examples/simple_classifier.py

Or with uvicorn:
    uvicorn examples.simple_classifier:app --reload
"""

import numpy as np
from fastmlapi import MLController, preprocessing, postprocessing


class SimpleClassifier(MLController):
    """
    A simple binary classifier for demonstration.
    
    In a real scenario, you would load an actual trained model
    (sklearn, pytorch, tensorflow, etc.)
    """
    
    model_name = "simple-classifier"
    model_version = "1.0.0"
    title = "Simple Classifier API"
    description = "A demo binary classifier API built with FastMLAPI"
    
    def load_model(self):
        """
        Load the ML model.
        
        Replace this with your actual model loading logic:
        - sklearn: joblib.load("model.pkl")
        - pytorch: torch.load("model.pt")
        - tensorflow: tf.keras.models.load_model("model.h5")
        """
        # Simple demo model: classify based on sum of features
        def simple_model(features):
            # Returns 1 if sum > 0, else 0
            return np.array([1 if np.sum(features) > 0 else 0])
        
        return simple_model
    
    @preprocessing
    def preprocess(self, data: dict) -> np.ndarray:
        """
        Transform raw input data into model-ready format.
        
        Args:
            data: Raw input from API request
            
        Returns:
            Preprocessed data ready for model prediction
        """
        features = data.get("features", [])
        
        # Convert to numpy array and reshape
        arr = np.array(features, dtype=np.float32)
        
        # Ensure 2D shape (batch_size, features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        
        return arr
    
    @postprocessing
    def postprocess(self, prediction: np.ndarray) -> dict:
        """
        Transform model output into API response format.
        
        Args:
            prediction: Raw model output
            
        Returns:
            JSON-serializable dictionary for API response
        """
        class_id = int(prediction[0])
        labels = {0: "negative", 1: "positive"}
        
        return {
            "class_id": class_id,
            "label": labels.get(class_id, "unknown"),
            "confidence": 0.95,  # Placeholder - real model would compute this
        }


# Create the controller instance
classifier = SimpleClassifier()

# Export the app for uvicorn
app = classifier.app


if __name__ == "__main__":
    # Run directly with python
    print("Starting Simple Classifier API...")
    print("API docs available at: http://localhost:8000/docs")
    classifier.run(host="0.0.0.0", port=8000)
