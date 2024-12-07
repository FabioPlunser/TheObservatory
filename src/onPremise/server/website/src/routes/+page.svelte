<script lang="ts">
  import { onMount } from "svelte";
  import Icon from "@iconify/svelte";

  import VideoStream from "$lib/components/VideoStream.svelte";

  let cameras: any[] = $state([]);
  let rooms: any[] = $state([]);
  let selectedRoom: string | null = null;
  onMount(async () => {
    await loadData();
    // Refresh camera status periodically
    setInterval(loadData, 5000);
  });

  async function loadData() {
    const [camerasRes, roomsRes] = await Promise.all([
      fetch("/api/get-cameras"),
      fetch("/api/get-rooms"),
    ]);
    cameras = await camerasRes.json();
    rooms = await roomsRes.json();
  }

  let filteredCameras = $derived.by(() => {
    selectedRoom ? cameras.filter((c) => c.room_id === selectedRoom) : cameras;
  });
  // -------------------------------------------------------------------
  // -------------------------------------------------------------------
  import { persistentStore } from "$lib/helper/store.svelte";
  const theme = persistentStore("theme", "light");
  // -------------------------------------------------------------------
  // -------------------------------------------------------------------

  let menuBarStates = $state(["Cameras", "Alerts", "Rooms"]);
  let currentState = $state("Cameras");

  async function deleteCamera(camera) {
    let res = await fetch(`/api/delete-camera/${camera.id}`, {
      method: "POST",
    });
    let data = await res.json();
    console.log(data);
    loadData();
  }

  $inspect({ cameras, rooms, selectedRoom, filteredCameras });
</script>

<div class="w-full px-2 py-2 fixed top-0">
  <div class="mx-auto flex items-center justify-between">
    <div>
      <h1 class="text-3xl font-bold dark:text-white">TheObservatory</h1>
    </div>

    <div>
      <label class="swap swap-rotate">
        <!-- this hidden checkbox controls the state -->
        <input
          type="checkbox"
          class="theme-controller"
          value="light"
          checked={theme.value === "dark"}
          onchange={(e) =>
            (theme.value = e.currentTarget.checked ? "dark" : "light")}
        />

        <!-- sun icon -->
        <svg
          class="swap-off h-10 w-10 fill-current"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
        >
          <path
            d="M5.64,17l-.71.71a1,1,0,0,0,0,1.41,1,1,0,0,0,1.41,0l.71-.71A1,1,0,0,0,5.64,17ZM5,12a1,1,0,0,0-1-1H3a1,1,0,0,0,0,2H4A1,1,0,0,0,5,12Zm7-7a1,1,0,0,0,1-1V3a1,1,0,0,0-2,0V4A1,1,0,0,0,12,5ZM5.64,7.05a1,1,0,0,0,.7.29,1,1,0,0,0,.71-.29,1,1,0,0,0,0-1.41l-.71-.71A1,1,0,0,0,4.93,6.34Zm12,.29a1,1,0,0,0,.7-.29l.71-.71a1,1,0,1,0-1.41-1.41L17,5.64a1,1,0,0,0,0,1.41A1,1,0,0,0,17.66,7.34ZM21,11H20a1,1,0,0,0,0,2h1a1,1,0,0,0,0-2Zm-9,8a1,1,0,0,0-1,1v1a1,1,0,0,0,2,0V20A1,1,0,0,0,12,19ZM18.36,17A1,1,0,0,0,17,18.36l.71.71a1,1,0,0,0,1.41,0,1,1,0,0,0,0-1.41ZM12,6.5A5.5,5.5,0,1,0,17.5,12,5.51,5.51,0,0,0,12,6.5Zm0,9A3.5,3.5,0,1,1,15.5,12,3.5,3.5,0,0,1,12,15.5Z"
          />
        </svg>

        <!-- moon icon -->
        <svg
          class="swap-on h-10 w-10 fill-current"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
        >
          <path
            d="M21.64,13a1,1,0,0,0-1.05-.14,8.05,8.05,0,0,1-3.37.73A8.15,8.15,0,0,1,9.08,5.49a8.59,8.59,0,0,1,.25-2A1,1,0,0,0,8,2.36,10.14,10.14,0,1,0,22,14.05,1,1,0,0,0,21.64,13Zm-9.5,6.69A8.14,8.14,0,0,1,7.08,5.22v.27A10.15,10.15,0,0,0,17.22,15.63a9.79,9.79,0,0,0,2.1-.22A8.11,8.11,0,0,1,12.14,19.73Z"
          />
        </svg>
      </label>
    </div>
  </div>
</div>

<!-- Menu in the center -->
<div class="fixed left-1/2 top-0 -translate-x-1/2 flex justify-center pt-2">
  <ul class="menu menu-horizontal bg-base-200 rounded-box dark:text-white">
    {#each menuBarStates as option}
      <li>
        <button
          onclick={() => (currentState = option)}
          class={currentState === option ? "active" : ""}>{option}</button
        >
      </li>
    {/each}
  </ul>
</div>

<div class="mt-24"></div>
<div class="mx-4">
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
    {#each cameras as camera (camera.id)}
      <div class="rounded-lg p-4 bg-base-200 shadow-lg">
        <div class="mb-2 flex justify-between items-center">
          <h1 class="text-lg font-bold">
            {camera.name || camera.id}
          </h1>
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
  </div>
</div>
