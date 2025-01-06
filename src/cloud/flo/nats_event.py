import json
import nats
import asyncio

from cloud import NATS_URL



async def trigger_upload_finished_event():
    # Connect to NATS
    nc = await nats.connect(NATS_URL)

    # Sample data to send
    camera_id = "camera_1"
    presigned_image_url = "https://the-observatory-faces-to-check.s3.us-east-1.amazonaws.com/pexels-olly.jpg"

    # Prepare the data
    event_data = {
        "cameraId": camera_id,
        "presignedImageUrl": presigned_image_url
    }

    # Publish the message to the 'upload.finished' subject
    await nc.publish("upload.finished", json.dumps(event_data).encode())
    print(f"Event sent: {event_data}")
    # Sample data to send
    camera_id = "camera_2"
    presigned_image_url = "https://the-observatory-faces-to-check.s3.us-east-1.amazonaws.com/PatrickStewart2004.jpg"

    # Prepare the data
    event_data = {
        "cameraId": camera_id,
        "presignedImageUrl": presigned_image_url
    }

    # Publish the message to the 'upload.finished' subject
    await nc.publish("upload.finished", json.dumps(event_data).encode())
    print(f"Event sent: {event_data}")

    # Close the connection
    await nc.close()


# Run the event trigger
if __name__ == "__main__":
    asyncio.run(trigger_upload_finished_event())
