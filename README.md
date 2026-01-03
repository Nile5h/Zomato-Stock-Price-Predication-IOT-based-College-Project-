Note: This code works reliably on Python 3.11 and lower because some of the libraries used (especially TensorFlow, Keras, and their dependencies) do not yet have stable support for Python 3.12+. The issue is library compatibility, not a problem in the code itself.

The system is divided into two main components:

Prediction Engine (LZomato_updated.py): A Deep Learning model (LSTM) that fetches historical data and predicts the next 7 days of stock price "High" and "Low" values.

Hardware Link (send_to_esp32.py): Since it is an IOT based project I have to add IOT component in it. A bridge script that reads the predictions and sends them to an external ESP32 device via a USB serial port (COM7).
