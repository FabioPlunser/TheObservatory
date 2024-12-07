<script lang="ts">
  import { onDestroy, onMount } from "svelte";

  let { cameraId } = $props();
  let canvas: HTMLCanvasElement;
  let websocket: WebSocket;
  let reconnectTimer: number;
  let isConnected = $state(false);

  onMount(() => {
    connectWebSocket();
  });

  function connectWebSocket() {
    clearTimeout(reconnectTimer);
    websocket?.close();

    // Connect to the viewer websocket endpoint
    websocket = new WebSocket(
      `ws://${window.location.host}/ws/view/${cameraId}`
    );

    const ctx = canvas.getContext("2d");

    websocket.onopen = () => {
      isConnected = true;
    };

    websocket.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      const img = new Image();
      img.onload = () => {
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
        ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      img.src = "data:image/jpeg;base64," + data.frame;
    };

    websocket.onclose = () => {
      isConnected = false;
      // Reconnect after 5 seconds
      reconnectTimer = setTimeout(connectWebSocket, 5000);
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      websocket.close();
    };
  }

  onDestroy(() => {
    clearTimeout(reconnectTimer);
    websocket?.close();
  });
</script>

<div class="relative w-full h-full">
  <canvas
    bind:this={canvas}
    width={640}
    height={480}
    class="w-full h-full object-cover"
  />
  {#if !isConnected}
    <div
      class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 text-white"
    >
      Reconnecting...
    </div>
  {/if}
</div>
