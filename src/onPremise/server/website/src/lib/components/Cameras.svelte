<script lang="ts">
  import { getContext, onMount, getAllContexts } from "svelte";
  import Icon from "@iconify/svelte";
  import VideoStream from "$lib/components/VideoStream.svelte";
  // -------------------------------------------------------------------
  // -------------------------------------------------------------------
  let cameras: any[] = $state([]);
  let rooms: any[] = $state([]);
  let selectedRoom: string | null = null;
  // -------------------------------------------------------------------
  // -------------------------------------------------------------------
  onMount(async () => {
    await loadData();
    // Refresh camera status periodically
    setInterval(loadData, 5000);
  });
  // -------------------------------------------------------------------
  // -------------------------------------------------------------------
  const SERVER_URL = import.meta.env.VITE_SERVER_URL;
  async function loadData() {
    const [camerasRes, roomsRes] = await Promise.all([
      fetch(`${SERVER_URL}/api/get-cameras`),
      fetch(`${SERVER_URL}/api/get-rooms`),
    ]);
    cameras = await camerasRes.json();
    rooms = await roomsRes.json();
  }
  // -------------------------------------------------------------------
  let filteredCameras = $derived.by(() => {
    selectedRoom ? cameras.filter((c) => c.room_id === selectedRoom) : cameras;
  });
  // -------------------------------------------------------------------
  // -------------------------------------------------------------------
  async function deleteCamera(camera) {
    let res = await fetch(`${SERVER_URL}/api/delete-camera/${camera.id}`, {
      method: "POST",
    });
    let data = await res.json();
    console.log(data);
    loadData();
  }
  // -------------------------------------------------------------------
  // -------------------------------------------------------------------
</script>

<div class="mx-4">
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
    {#if cameras.length > 0}
      {#each cameras as camera (camera.id)}
        <div class="rounded-lg p-4 bg-base-200 shadow-lg">
          <div class="mb-2 flex justify-between items-center">
            <div>
              <h1 class="text-lg font-bold">
                {camera.name}
              </h1>
              <p>Id: {camera.id}</p>
            </div>
            <button
              onclick={async () => {
                await deleteCamera(camera);
                loadData();
              }}
              ><Icon
                icon="material-symbols:delete"
                width="24"
                height="24"
              /></button
            >
          </div>

          <div class="aspect-vide rounded overflow-hidden">
            <VideoStream cameraId={camera.id} />
          </div>

          <div class="mt-2 text-sm">
            <p>
              Room: {rooms.find((r) => r.id === camera.room_id)?.name ||
                "Unassigned"}
            </p>
            {#if camera.status === "offline"}
              <div class="badge badge-error">{camera.status}</div>
            {/if}
            {#if camera.status === "online" || camera.status === "registered"}
              <div class="badge badge-success">{camera.status}</div>
            {/if}
            <div class="badge badge-primary">{camera.last_seen}</div>
          </div>
        </div>
      {/each}
    {:else}
      <h1 class="flex justify-center mx-auto font-bold text-2xl">
        No cameras connected
      </h1>
    {/if}
  </div>
</div>
