<script lang="ts">
  import "../app.css";
  import Nav from "$lib/components/Nav.svelte";
  import { onMount, setContext } from "svelte";
  import { getState } from "$lib/helper/globalState.svelte";
  import { getCompany, update_cloud_url } from "$lib/helper/fetches";
  import AlarmPopup from "$lib/components/AlarmPopup.svelte";

  let { children } = $props();

  let company = getState<Company | undefined>("company", undefined);
  let alarms = $state([]);

  const SERVER_URL = import.meta.env.VITE_SERVER_URL;

  let url = $state();
  async function update_url(e) {
    e.preventDefault();
    let res = await update_cloud_url(url);
    let data = await res.json();
    await getCompany();
    location.reload();
  }
  onMount(async () => {
    let res = await getCompany();
    if (!res.ok) {
      company.value = undefined;
      alert("Invalid URL");
      return;
    }
    let data = await res.json();
    if (data?.company) {
      company.value = data.company;
    }
    // setInterval(getActiveAlarm, 1000);
  });

  // async function getActiveAlarm() {
  //   let res = await fetch(`${SERVER_URL}/api/alarm`);
  //   let data = await res.json();
  //   alarms = data.alarms;
  // }

  async function dismissAlarm() {
    try {
      const res = await fetch(`${SERVER_URL}/alarm/reset`, {
        method: "POST",
      });
      if (res.ok) {
        alarms = [];
      }
    } catch (error) {
      console.error("Error dismissing alarm:", error);
    }
  }
</script>

<main class="mx-4">
  {#if alarms && alarms.length > 0}
    <AlarmPopup {alarms} />
  {/if}
  <!-- {#if company.value && !company.value.cloud_url}
    <div
      class="flex justify-center mx-auto my-auto h-screen pt-24 gap-4 text-white"
    >
      <div class="w-min">
        <input
          class="input input-bordered"
          type="text"
          placeholder="Enter your cloud url"
          bind:value={url}
        />
        <p class="w-full font-bold mx-auto">
          Pure domain name example: theobservatory.com or just ip address:
          54.167.70.96:4222
        </p>
      </div>
      <button onclick={update_url} class="btn btn-primary">Submit</button>
    </div>
  {:else if !company}
    <h1>Unforseen Error</h1>
  {:else} -->
    <Nav />
    <div class="mt-24"></div>
    {@render children()}
  <!-- {/if} -->
</main>
