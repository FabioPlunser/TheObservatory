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
```bash 
cd src/onPremis 
./run.sh
```
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