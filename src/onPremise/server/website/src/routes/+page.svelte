<script lang="ts">
  import { persistentState } from "$lib/helper/persistentState.svelte";
  import { getPages } from "$lib/helper/pages.svelte";

  const Thing = $derived.by(() => {
    let currentPage = persistentState<currentPageType>("currentPage");
    let pages = getPages().value;
    return pages.find((p) => p.name === currentPage.value.name)?.component;
  });
</script>

{#if Thing}
  <Thing></Thing>
{/if}
