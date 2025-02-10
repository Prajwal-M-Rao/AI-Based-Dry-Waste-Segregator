									AI-based Dry Waste Segregator

This project is an AI-powered Dry Waste Segregator designed to classify different types of dry waste using machine learning. It features real-time data visualization through a web interface, making waste management more efficient.

Features:
	Automated waste classification.
	Real-time monitoring on a web-based interface.
	Efficient segregation using MobileNet and servo motors.

1.Prerequisites:
	Hardware: Raspberry Pi, camera module, servo motor, DC motor, conveyor belt.
	Software: Python 3, TensorFlow, OpenCV, RPi.GPIO.- Python 3.x, Flask, Firebase Admin SDK, Chart.js for visualization, Firestore database for data storage

2. Hardware Setup:
	Construct a conveyor belt system driven by a DC motor. The camera module should be installed above the conveyor belt to capture images of waste moving along it. A servo motor is placed at the end of the conveyor belt to 	enable waste classification and sorting by rotating to direct waste into the respective bins.

	Connections
		Connect the camera module to the Raspberry Pi via the CSI port using a flex cable.
		Use the GPIO pins on the Raspberry Pi to connect the servo motor. The pin configuration is as follows:
		- Pin 2 (5V): Power supply for the servo motor.
		- Pin 6 (GND): Ground connection shared between the Raspberry Pi and the servo motor.
		- Pin 11 (GPIO 17): Provides the PWM signal to control the servo motor's position.

	Dependencies Installation
		Install all necessary dependencies on the Raspberry Pi as per the requirements of the model. This includes libraries for image processing, machine learning, and hardware control (e.g., OpenCV, TensorFlow, 		RPi.GPIO, etc.).

	Code Implementation
		Create a new Python file for the project.
		Load the pre-trained MobileNet model for image classification.
		Integrate the code with the required APIs to establish a web connection. Follow the steps outlined in the web connection process to ensure proper integration

	Working Mechanism
		The camera module captures images of waste items on the conveyor belt.
		The images are classified using the MobileNet model, identifying the type of waste.
		The conveyor belt transports the classified waste to the end of the system.
		Based on the classification result, the servo motor rotates to direct the waste into the appropriate bin.
		The classification result, including the waste category and accuracy, is updated on a web-based interface for real-time monitoring.

	Note : Copy paste the code that has been saved as raspberry pi code in the pi which you are using

3. Code Explaination:
	This project uses MobileNetV2 to classify waste into categories like glass, paper, plastic, and metal. It includes dataset preprocessing, model training, fine-tuning, and evaluation.

	 Requirements: Python , TensorFlow, Matplotlib, VS Code
	 Install all dependencies

	 Steps : 
		Preprocess Data: Augment data (rotation, zoom, flipping) and split into training/validation sets.
		Build Model: Use MobileNetV2 with custom layers (GlobalAveragePooling, Dense, Dropout).
		Train Model: Perform initial training (frozen base model) and fine-tune selected layers.
		Evaluate: Test accuracy and plot training graphs for loss and accuracy.
	Usage:
		Clone the repo and prepare your dataset.
		Train the model: The trained model is saved as waste_classification_model.h5.
		Use the model:
			from tensorflow.keras.models import load_model
			model = load_model('waste_classification_model.h5')
	Results:
		Test Accuracy
		Visualization: Loss and accuracy plots during training.

4. Firebase Firestore setup:
	Add a Firebase project.
	Generate a service account key JSON file and provide its path in the credentials.Certificate() method.

	Configure Firebase
		Create a Firestore database and add a collection named waste_data.
		Download the Firebase service account key and update the path in the script.
	
	Load the MobileNet Model
		Use TensorFlow to load the pre-trained MobileNet model for waste type detection.

	Installation 
		Configure Firebase:
	 		Obtain the Firebase Admin SDK JSON file from your Firebase project.
	 		Save it as firebase_credentials.json in the project directory.
		Run the application:
	 		python app.py

		Access the web interface:
		 	Open a browser and navigate to http://127.0.0.1:5000.

		Verify features:
	 	 	Test waste data visualization, weekly analysis, and data export functionality.
5. Server Setup: 
	We have set up a server for image detection and classification since the Raspberry Pi cannot handle the processing load alone. The captured image is sent from the Pi to a connected system, where detection and classification take place. The processed results are then sent back to the Pi for further processing.

6. How to Run the Project: 
	Step-by-step instructions to start and operate the system, including:
		- Powering the Raspberry Pi.
		- Running the Python script.
		- Connecting the server for image classification
		- Host the web as per the instructions given above.
		- Access the web interface for monitoring results.


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Credits:

Prof. Padma Prasada
Assistant Professor
SDMIT, Ujire
Guide of this project

Created by:

Akshaj M (USN: 4SU21AD005)
Bilvarchan Salanki M (USN: 4SU21AD012)
Gagan V (USN: 4SU21AD018)
Prajwal M (USN: 4SU21AD035)

If any queries feel free to contact:
raoprajwalm@gmail.com