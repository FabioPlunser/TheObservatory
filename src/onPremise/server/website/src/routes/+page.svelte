<script lang="ts">
  import { getState } from "$lib/helper/globalState.svelte";

  let currentPage = getState<currentPageType>("currentPage");

  let show = $derived(currentPage.value);
</script>

<show.component />
