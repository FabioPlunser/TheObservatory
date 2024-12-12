<script lang="ts">
  import "../app.css";
  import Nav from "$lib/components/Nav.svelte";
  import { setContext } from "svelte";
  let { children } = $props();
</script>

<main class="dark:text-white">
  <Nav />
  <div class="mt-24"></div>
  {@render children()}
</main>
