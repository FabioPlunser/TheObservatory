<script lang="ts">
  import { onMount } from "svelte";
  import Icon from "@iconify/svelte";

  let alarms: any = $state();

  const SERVER_URL = import.meta.env.VITE_SERVER_URL;

  async function loadData() {
    console.log("Get rooms");
    const res = await fetch(`${SERVER_URL}/api/get-alarms`);
    console.log(res);
    if (!res.ok) {
      throw new Error("Failed to fetch alarms");
    }
    const data = await res.json();
    alarms = data;
  }

  onMount(async () => {
    await loadData();
  });

  async function activateAlarm(alarm) {
    const res = await fetch(
      `${SERVER_URL}/api/activate-alarm?alarm_id=${alarm.id}`,
      {
        method: "POST",
      }
    );
    if (!res.ok) {
      throw new Error("Failed to activate alarm");
    }
    const data = await res.json();
    console.log(data);
  }
  $inspect(alarms);
</script>

<div>
  {#if alarms && alarms.length > 0}
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {#each alarms as alarm}
        <div class="rounded-lg p-4 bg-base-200 shadow-lg relative">
          <div class="">
            <Icon
              class="text-white"
              icon="material-symbols:detector-alarm"
              width="24"
              height="24"
            />
            <div class="text-white font-bold flex">
              Alarm: {alarm.id}
            </div>
            <div class="absolute right-0 p-2 top-0 text-red-500">
              <Icon icon="material-symbols:delete" width="24" height="24" />
            </div>
          </div>
          <div class="flex flex-col gap-2">
            {#if alarm.active}
              <h1 class="badge badge-success">Active: {alarm.active}</h1>
            {:else}
              <h1 class="badge badge-error">Active: {alarm.active}</h1>
            {/if}
            <h1 class="badge badge-primary">Last_Seen: {alarm.last_seen}</h1>
            <h1 class="badge badge-info">Status: {alarm.status}</h1>
            {#if alarm.active}
              <button class="btn btn-info">Disable</button>
            {:else}
              <button
                onclick={() => activateAlarm(alarm)}
                class="btn btn-success">Enable</button
              >
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <p>No alarms found</p>
  {/if}
</div>
