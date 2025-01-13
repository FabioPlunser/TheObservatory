<script lang="ts">
  import { onMount } from "svelte";
  import VideoStream from "./VideoStream.svelte";
  import Icon from "@iconify/svelte";

  let { alarms } = $props();
  let isBlinking = $state(true);

  // Set up blinking effect
  let interval: number;
  onMount(() => {
    interval = setInterval(() => {
      isBlinking = !isBlinking;
    }, 500);

    return () => {
      if (interval) clearInterval(interval);
    };
  });

  const SERVER_URL = import.meta.env.VITE_SERVER_URL;
  async function dismissAlarm() {
    let res = await fetch(`${SERVER_URL}/api/alarm/reset`, {
      method: "POST",
    });

    if (res.ok) {
      alarms = [];
    }
  }
</script>

<div class="fixed inset-0 z-50">
  <!-- Blinking border container -->
  <div
    class={`fixed inset-0 border-8 transition-colors duration-500 
    ${isBlinking ? "border-red-600" : "border-transparent"}`}
  >
    <!-- Overlay -->
    <div class="absolute inset-0 bg-black/75 backdrop-blur-sm">
      <!-- Content container -->
      <div class="container mx-auto p-6 mt-10">
        <div class="bg-gray-900 rounded-xl p-6">
          <!-- Header with dismiss button -->
          <div class="w-full">
            <h2 class="text-2xl font-bold">
              ⚠️ Security Alert - Unknown Persons Detected
            </h2>
            <div class="flex justify-center">
              <button onclick={dismissAlarm} class="btn btn-error">
                Dismiss
              </button>
            </div>
          </div>

          <!-- Grid of camera feeds -->
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {#each alarms as alert (alert.camera_id)}
              <div class="bg-gray-800 rounded-lg p-4 shadow-lg">
                <!-- Video stream -->
                <div
                  class="aspect-video mb-4 bg-black rounded-lg overflow-hidden"
                >
                  <VideoStream cameraId={alert.camera_id} />
                </div>

                <!-- Alert details -->
                <div class="flex gap-4">
                  <!-- Unknown face image -->
                  <div class="w-32 h-32 rounded overflow-hidden flex-shrink-0">
                    <img
                      src={alert.unknown_face_url}
                      alt="Unknown person detected"
                      class="w-full h-full object-cover"
                    />
                  </div>

                  <!-- Alert info -->
                  <div>
                    <p class="font-semibold">Camera ID: {alert.camera_id}</p>
                    <p class="text-sm text-gray-400">
                      Detected: {new Date(alert.timestamp).toLocaleString()}
                    </p>
                    <div class="mt-2">
                      <span
                        class="px-2 py-1 bg-red-500/20 text-red-400 text-sm rounded-full"
                      >
                        Active Alert
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
