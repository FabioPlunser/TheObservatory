# TheObservatory
Distriputed Systems Project


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