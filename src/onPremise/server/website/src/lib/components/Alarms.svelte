<script lang="ts">
  import { onMount } from "svelte";

  let rooms: any = $state();

  async function loadData() {
    console.log("Get rooms");
    const res = await fetch("/api/get-alarms");
    const data = await res.json();
    rooms = data;
  }

  $inspect(rooms);
</script>

<div>
  <h1>Alarms</h1>
  {#each rooms as room}
    {JSON.stringify(room)}
  {/each}
</div>
