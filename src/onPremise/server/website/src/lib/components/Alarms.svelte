<script lang="ts">
  import { onMount } from "svelte";
  onMount(async () => {
    await loadData();
  });

  let rooms: any = $state();

  async function loadData() {
    const res = await fetch("/api/get-alarms");
    const data = await res.json();
    rooms = data;
  }
</script>

<div>
  {#each rooms as room}
    {JSON.stringify(room)}
  {/each}
</div>
