<script lang="ts">
  import { onMount } from "svelte";
  import Icon from "@iconify/svelte";

  let alarms: any = $state();

  const SERVER_URL = import.meta.env.VITE_SERVER_URL;

  async function loadData() {
    const res = await fetch(`${SERVER_URL}/api/alarms`);
    if (!res.ok) {
      throw new Error("Failed to fetch alarms");
    }
    const data = await res.json();
    alarms = data;
    console.log(alarms);
  }

  onMount(async () => {
    await loadData();
    setInterval(loadData, 5000);
  });

  async function activateAlarm(alarm) {
    const res = await fetch(
      `${SERVER_URL}/api/alarm/enable?alarm_id=${alarm.id}`,
      {
        method: "POST",
      }
    );
    if (!res.ok) {
      throw new Error("Failed to activate alarm");
    }
    const data = await res.json();
    loadData();
  }

  async function deactivateAlarm(alarm) {
    const res = await fetch(
      `${SERVER_URL}/api/alarm/disable?alarm_id=${alarm.id}`,
      {
        method: "POST",
      }
    );
    if (!res.ok) {
      throw new Error("Failed to deactivate alarm");
    }
    const data = await res.json();
    loadData();
  }

  async function deleteAlarm(alarm) {
    const res = await fetch(
      `${SERVER_URL}/api/alarm/delete?alarm_id=${alarm.id}`,
      {
        method: "POST",
      }
    );
    if (!res.ok) {
      throw new Error("Failed to delete alarm");
    }
    const data = await res.json();
    loadData();
  }
</script>

<div>
  {#if alarms && alarms.length > 0}
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {#each alarms as alarm}
        <div class="rounded-lg p-4 bg-base-200 shadow-lg relative">
          <div class="">
            <Icon
              icon="material-symbols:detector-alarm"
              width="24"
              height="24"
            />
            <div class="font-bold flex">
              Alarm: {alarm.id}
            </div>
            <div class="absolute right-0 p-2 top-0 text-red-500">
              <button onclick={() => deleteAlarm(alarm)}>
                <Icon icon="material-symbols:delete" width="24" height="24" />
              </button>
            </div>
          </div>
          <div class="flex flex-col gap-2">
            {#if alarm.active}
              <h1 class="badge badge-success">Active</h1>
            {:else}
              <h1 class="badge badge-error">Not Active</h1>
            {/if}
            <h1 class="badge badge-primary">Last_Seen: {alarm.last_seen}</h1>
            {#if alarm.connected}
              <h1 class="badge badge-success">Connected</h1>
            {:else}
              <h1 class="badge badge-error">Disconnected</h1>
            {/if}
            {#if alarm.active}
              <button
                onclick={() => deactivateAlarm(alarm)}
                class="btn btn-error">Disable</button
              >
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
    <h1 class="flex justify-center text-2xl font-bold items-center">
      No alarms found
    </h1>
  {/if}
</div>
