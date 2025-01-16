# TheObservatory
Distriputed Systems Project

## Project Description 

## On Premise 
The on premise part is a server responsible to accumulate all camera streams inside of a company and 
comunicate with the cloud for facial recognition. 
The on premmise server handels the following 
- Registration of cameras and saving to db 
- Registration of alarms and saving to db 
- Website to view the camera streams and other features
- Uploading images to the cloud for facial recognition 
- Receiving the results from the cloud and sending alarms to the alarms 
- Detecting persons in the camera streams with YOLO and sending the images to the cloud for facial recognition 

To run the On Premis server do the following:
(linux) 
```bash 
cd src/onPremis 
./run.sh
```
(windows)
```bash
cd src/onPremis
run.ps1
```
For the website to work a .env file is required in the src\onPremise\server\website\ folder with the following content:
```bash
VITE_SERVER_URL=http://localhost:8000
```

The start script should create a virtual environment and install all the required packages.
It also checks if the aws credentials are set up correctly and if they are valid.
When the server is started you will need to input the number of cameras you want to simulate(Webcam feed)
after that the number of Alarms you want to simulate. And if you want to us the 6 simulated camera feed 
from the wiseNet Dataset(src\data\video_sets\...).

Python 3.12 is required to run the project.

This will do everything automatically for you.

Or else for `deveolpment` you can do the following: 

To run the project its recommended to use an virtual environment. To create a virtual environment run the following command:
```bash
python3 -m venv venv
```
To activate the virtual environment run the following command:
```bash
source venv/bin/activate
```
To install the required packages run the following command:
```bash
pip install -r requirements.txt
```
Bun or node-js is required to run the website.
For Bun:
```bash
bun install
bun run build
```
For node-js:
```bash
npm install
npm run build
```
To start the website run the following command:
```bash
npm run preview
```
or
```bash
bun run preview
```
After starting the web server, proceed with setting up the server.

To start the server run the following command:
In src/onPremise/devices/server
```bash
python edge_server.py
```
To start the webcam of your device run the following command:
In src/onPremise/devices/emulated
```bash
python camera.py
```
If you want to simulate cameras run the following command:
In src/onPremise/devices/emulated
```bash
python simulate_cameras.py
```
You will receive a web address to access the website, when starting one of the camera methods above.
```