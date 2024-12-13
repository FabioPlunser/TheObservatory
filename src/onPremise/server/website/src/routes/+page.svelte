<script lang="ts">
  import { persistentState } from "$lib/helper/persistentState.svelte";
  import { getState } from "$lib/helper/globalState.svelte";
  import { onMount } from "svelte";

  // let currentPage = getState<currentPageType>("currentPage");
  let currentPage = persistentState<currentPageType>("currentPage");
  onMount(() => {
    currentPage = persistentState<currentPageType>("currentPage");1
  });
  let show = $derived(currentPage.value);

  $inspect(show);
</script>

{#if show}
  <show.component />
{/if}
