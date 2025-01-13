<script lang="ts">
  import { error } from "@sveltejs/kit";
  import { onMount, onDestroy } from "svelte";

  let { cameraId } = $props();
  let imgElement: HTMLImageElement | null = null;
  let ws: WebSocket;
  let isConnected = $state(false);

  function connectWebSocket() {
    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    const wsURL = `${wsProtocol}://${import.meta.env.VITE_SERVER_URL.replace("http://", "")}/ws/camera/${cameraId}`;

    ws = new WebSocket(wsURL);

    ws.binaryType = "arraybuffer"; // Important!

    ws.onmessage = function (event) {
      const blob = new Blob([event.data], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      if (imgElement) {
        imgElement.onload = () => {
          URL.revokeObjectURL(url); // Revoke only after the image has loaded
        };
        imgElement.src = url;
      }
    };

    ws.onopen = () => {
      isConnected = true;
    };

    ws.onerror = (error) => {
      isConnected = false;
      setTimeout(connectWebSocket, 100);
    };

    ws.onclose = () => {
      isConnected = false;
      setTimeout(connectWebSocket, 100);
    };
  }

  function cleanup() {
    ws.close();
  }

  onMount(() => {
    connectWebSocket();
  });

  onDestroy(() => {
    cleanup();
  });
</script>

<div class="relative w-full h-full">
  <!-- svelte-ignore a11y_media_has_caption -->
  <!-- svelte-ignore element_invalid_self_closing_tag -->
  <!-- svelte-ignore a11y_missing_attribute -->
  <img bind:this={imgElement} class="w-full h-full object-cover" />
  {#if !isConnected}
    <div
      class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50"
    >
      <p class="">Connecting...</p>
    </div>
  {/if}
</div>
