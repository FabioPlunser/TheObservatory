<script lang="ts">
  import { onMount, onDestroy } from "svelte";

  let { cameraId } = $props();
  let videoElement: HTMLVideoElement;
  let peerConnection: RTCPeerConnection;

  async function startStream() {
    try {
      // Create peer connection
      peerConnection = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      });

      // Handle incoming track
      peerConnection.ontrack = (event) => {
        if (videoElement) {
          videoElement.srcObject = event.streams[0];
        }
      };

      // Create offer
      const offer = await peerConnection.createOffer({
        offerToReceiveVideo: true,
      });
      await peerConnection.setLocalDescription(offer);

      // Send offer to server
      const response = await fetch(`/webrtc/${cameraId}/offer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
        }),
      });

      const answer = await response.json();
      await peerConnection.setRemoteDescription(answer);
    } catch (e) {
      console.error("Error starting stream:", e);
    }
  }

  async function stopStream() {
    if (peerConnection) {
      peerConnection.close();
      await fetch(`/webrtc/${cameraId}/close`, { method: "POST" });
    }
    if (videoElement) {
      videoElement.srcObject = null;
    }
  }

  onMount(() => {
    startStream();
  });

  onDestroy(() => {
    stopStream();
  });
</script>

<div class="relative w-full h-full">
  <!-- svelte-ignore a11y_media_has_caption -->
  <!-- svelte-ignore element_invalid_self_closing_tag -->
  <video
    bind:this={videoElement}
    autoplay
    playsinline
    class="w-full h-full object-cover"
  />
</div>
