# import serial
# import pandas as pd
# import time

# try:
#     ser = serial.Serial("COM7",115200)
#     df = pd.read_csv("next7days_high_low.csv")

#     for _, row in df.iterrows():
#         # line = f"{row['Date']},{row['Predicted_Close']:.1f}\r\n"
#         line = f"{row['Date']},{row['Predicted_Low']:.1f} - {row['Predicted_High']:.1f}\n"
#         # print(line,type(row["Date"]))
#         ser.write(line.encode())
#         time.sleep(2)  # Wait for ESP to display
# except serial.serialutil.SerialException as sr:
#     print("Couldn't find the ESP32 please check the port it is connected. \n OR there must be another app communicating with the port.")

# print("END Code")


# --------------------------------------------------

# df = pd.read_csv("next7days_high_low.csv")

# for _, row in df.iterrows():
#         line = f"{row['Date']},{row['Predicted_Low']:.1f} - {row['Predicted_High']:.1f}\n"
#         print(line)
    
# --------------------------------------------


import serial, pandas as pd, time

ser = serial.Serial("COM7", 115200)
df = pd.read_csv("next7days_high_low.csv")

for _, row in df.iterrows():
    line = f"{row['Date']},{row['Predicted_Low']:.1f}-{row['Predicted_High']:.1f}\n"
    print(line)
    ser.write(line.encode())
    time.sleep(4)  # Wait for ESP to display
    