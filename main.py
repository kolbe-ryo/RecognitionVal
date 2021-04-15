import time
import datetime
import pandas as pd
import traceback
import cv2
import csv

import camera
import recognitionValueMeter

def main():

    while True:
        try:
            # Initialization
            data, df, series = [], [], []
            
            # Take picture and get time
            # imgOrg = camera.capture()
            imgOrg = "Please set your picture directly and name"
            imgOrg = cv2.imread(imgOrg)
            timeTakePic = str(datetime.datetime.now().strftime('%Y%m%d%H%M'))

            # Recognition value on steam meter
            recognitionValue, log = recognitionValueMeter.valueRecognition(imgOrg)

            # Save in csv
            print(timeTakePic, recognitionValue)
            data.append(timeTakePic)
            data.append(recognitionValue)
            data.append(log)

            with open('log.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data)
                f.close()

            # Time interval treatment
            print("Wait for a minute")
            time.sleep(60)

        except:
            traceback.print_exc()

if __name__ == "__main__":
    main()
