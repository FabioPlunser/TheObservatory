<script lang="ts">
  import { update_cloud_url } from "$lib/helper/fetches";
  import { getState } from "$lib/helper/globalState.svelte";

  let company = getState<Company | undefined>("company", undefined);
  async function updateURL() {
    let res = await update_cloud_url(company.value?.cloud_url);
  }
</script>

<div class="mx-4">
  {#if company.value}
    <div class="flex justify-center mx-auto">
      <div class="flex flex-col gap-4">
        <div>
          <h1 class="font-bold">ID: {company.value.id}</h1>
        </div>
        <label class="font-bold input input-bordered flex items-center gap-2">
          CLOUD_URL:
          <input
            id="name"
            bind:value={company.value.cloud_url}
            class="font-normal"
            placeholder="Enter your cloud url or ip address"
          />
        </label>
        <button class="btn btn-primary flex justify-center" onclick={updateURL}
          >Save Changes</button
        >
      </div>
    </div>
  {:else}
    <h1>No company found</h1>
  {/if}
</div>
