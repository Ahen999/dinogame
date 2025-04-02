# dinogame
Hand gesture control integration on the chrome dino game
This project examines the application of real-time hand gesture recognition to manage the Chrome Dino game, substituting conventional keyboard inputs with an AI-driven touchless interface. By utilizing computer vision, deep learning, and automation, the system identifies hand gestures through a webcam, analyzes them, and translates them into game commands. The main objective is to deliver a fluid, interactive, and accessible gaming experience, especially for those with physical disabilities who might find traditional input methods challenging.  
The project adheres to a systematic workflow that starts with data collection and preprocessing. A customized dataset of hand gestures was captured and enhanced using grayscale conversion, resizing, thresholding, and feature extraction methods. A Convolutional Neural Network (CNN) was subsequently trained to recognize various hand gestures, achieving 94.8% accuracy on the evaluation dataset. Advanced methodologies such as convex hull detection and convexity defect analysis were implemented to distinguish gestures based on the number of fingers extended. After training, the model was incorporated into a real-time system that processes camera input, identifies hand gestures, and translates them into game controls with the help of PyAutoGUI.  
To improve the user experience, a Graphical User Interface (GUI) was created using PyQt5, showing both the real-time webcam feed and the Chrome Dino game window. This configuration enables users to visually verify their gestures, promoting smooth and intuitive gameplay. The system was optimized to operate at 30 frames per second (FPS), ensuring minimal latency (<200ms) between gesture recognition and the game’s response.  
The outcomes illustrate the efficacy of gesture-based game control, with the system successfully identifying and executing game actions with high precision. The model performed well under various lighting conditions, although slight modifications were necessary to enhance performance in low-light settings. The GUI offered real-time feedback, permitting users to refine their gestures for optimal recognition.  
Despite its achievements, the project encountered several challenges, including background noise interference, variations in lighting, and limitations of CPU-based processing. Some gestures, especially those involving closely spaced fingers, were susceptible to misclassification due to overlapping contours. Prospective enhancements could include the implementation of Vision Transformers (ViTs) or MobileNet to boost model accuracy and efficiency. Additional improvements could aim at adaptive gesture tracking, allowing the system to adjust to changing hand positions and backgrounds dynamically.  
Beyond gaming, this project showcases the potential of gesture-based human-computer interaction in diverse domains, such as assistive technology, smart home automation, and virtual reality applications. By offering an intuitive, touch-free interface, it broadens the possibilities for hands-free interaction in both gaming and practical applications.  
In summary, this project successfully merges deep learning and computer vision to develop a gesture-based control system, validating that AI-powered gesture recognition can act as a feasible alternative to conventional input methods. While some challenges remain, continuous progress in AI and computer vision will likely strengthen the reliability and scalability of such systems, making them more pertinent in a touchless digital landscape. This project establishes a foundation for future exploration and advancement in gesture-based AI systems, presenting new avenues for accessibility and human-computer interaction. 











