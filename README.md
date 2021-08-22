# cardio
### Cardiovascular Monitoring and Detection System

#### System designed to detect incoming ECG signals from a connected IoT device, process the signals and then classify them as normal or abnormal after which the results are pushed to a mobile app.

The system was designed for real time analysis and deployed using the Google Cloud Platform. The system is triggered when it detects an incoming signal and then proceeds to:
  1) Filter the incoming signal to remove noise
  2) Uses QRS peak detection method to slice the signal into individual beats
  3) Performs beatwise classification
  4) Pushes the results to a connected mobile app for the user to see

A Long-Short Term Memory Recuurent Neural Network was designed to classify the signal as normal/abnormal. The system also calculates the average heart rate using the signal and pushes that information along with the result of the classifier to the user via a mobile app


#### Built Using:
#### Python

#### Libraries:


