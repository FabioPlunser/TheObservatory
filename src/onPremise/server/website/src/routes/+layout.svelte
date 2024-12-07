<script lang="ts">
	import '../app.css';
	let { children } = $props();
</script>

<main class="dark:text-white">
{@render children()}
</main>
