For the setup of the IoT devices, using a phone do the following:
1. Download a app which is able to transmit data via RTMP protocol. For example, IP Webcam on Android.
2. On aws create a repository via ECS
3. Use the instructions provided by aws to build and push the image to the repository
4. Register the task-definition.json with "aws ecs register-task-definition --cli-input-json file://task-definition.json"   
5. On aws create a cluster with the following settings:
    - Custer name: Your choice 
    - Infrastructure: AWS Fargate
6. In the create a Service with the following settings:
   - Launch type: Fargate
   - Task Definition(Family): The task definition you registered
   - Service name: Your choice
   - Number of tasks: 1(For now)
   - For Network configuration, select the VPC and Subnets you want to use
   - For the Security group, create a new one with the following settings:
        - Security group name: Your choice
        - Description: Your choice
        - VPC: The VPC you selected
        - Inbound rules: Add a rule with the following settings:
            - Type: Custom TCP
            - Protocol: TCP
            - Port range: 1935
            - Source: Anywhere(For now need to be looked into, security risk?)
7. After the service is created, now go to the cluster and click on the service you created and click on the task
8. Click on the network interface and copy the public IP address
9.  Update the public IP address in the IoT device code(./src/iot/iot.py) and run the code
10. Open the app you downloaded and enter the public IP address in the settings(rtmp://<public-url>:1935/live/stream7) and start the stream
11. Now a new folder should be created in the root directory with the name "output_frames" and the frames should be saved there
If use only want to use the WiseNet video feed, do the following:
1. Step 2-9 from above.
2. Copy the data from WiseNet like following: TheObservatory\data\video_sets\set_X\videoX_Y.avi
   1. The dataset should have 11 sets with 5 videos in set 1-4, and 6 videos in set 5-11.
3. Start the iot.py script(To kill the script close the terminal, ctrl+c does not work) ffmpeg is required to run the script
4. Start the edge.py script(To kill the script close the terminal, ctrl+c does not work)
5. The frames should be saved in the output_frames folder

If you running the docker container locally replace HOST_ADDR in iot.py and edge.py with "localhost".
Also start the docker container with the following command:
docker run -p 1935:1935 <Name of Docker image>