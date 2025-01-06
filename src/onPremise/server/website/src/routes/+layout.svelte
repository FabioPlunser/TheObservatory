<script lang="ts">
  import "../app.css";
  import Nav from "$lib/components/Nav.svelte";
  import { onMount, setContext } from "svelte";
  import { getState } from "$lib/helper/globalState.svelte";
  import { getCompany, update_cloud_url } from "$lib/helper/fetches";

  let { children } = $props();

  let company = getState<Company | undefined>("company", undefined);

  const SERVER_URL = import.meta.env.VITE_SERVER_URL;
  console.log(SERVER_URL);

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
    let data = await res.json();
    if (data?.company) {
      console.log(data.company);
      company.value = data.company;
    }
  });
</script>

<main class="dark:text-white">
  {#if company.value && !company.value.cloud_url}
    <div class="flex justify-center mx-auto my-auto h-screen pt-24 gap-4">
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
  {:else}
    <Nav />
    <div class="mt-24"></div>
    {@render children()}
  {/if}
</main>
