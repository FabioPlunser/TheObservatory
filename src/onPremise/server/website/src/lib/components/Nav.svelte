<script lang="ts">
  import Cameras from "./Cameras.svelte";
  import Alarms from "./Alarms.svelte";
  import Rooms from "./Rooms.svelte";
  import KnownFaces from "./KnownFaces.svelte";
  import Settings from "./Settings.svelte";
  // -------------------------------------------------------------------
  // -------------------------------------------------------------------
  import Icon from "@iconify/svelte";
  import { persistentStore } from "$lib/helper/store.svelte";
  import { getState } from "$lib/helper/globalState.svelte";
  import { persistentState } from "$lib/helper/persistentState.svelte";
  import { getPages } from "$lib/helper/pages.svelte";
  // -------------------------------------------------------------------
  // -------------------------------------------------------------------
  const theme = persistentStore("darkMode", "false");
  let currentPage = persistentState("currentPage", {
    name: "Cameras",
    component: Cameras,
  });
  let menuBarStates = getPages()?.value;
  // -------------------------------------------------------------------
  // -------------------------------------------------------------------
  function updateTheme(v) {
    theme.value = v;
    document.documentElement.classList.toggle("dark", !theme.value);
  }
</script>

<div class="w-full px-2 py-2 fixed top-0">
  <div class="mx-auto flex items-center justify-between">
    <div>
      <h1 class="text-3xl font-bold">TheObservatory</h1>
    </div>

    <div class="pr-4">
      <label class="swap swap-rotate">
        <!-- this hidden checkbox controls the state -->
        <input
          type="checkbox"
          class="theme-controller"
          value="light"
          bind:checked={() => theme.value, updateTheme}
        />

        <!-- sun icon -->
        <svg
          class="swap-off h-10 w-10 fill-current"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
        >
          <path
            d="M5.64,17l-.71.71a1,1,0,0,0,0,1.41,1,1,0,0,0,1.41,0l.71-.71A1,1,0,0,0,5.64,17ZM5,12a1,1,0,0,0-1-1H3a1,1,0,0,0,0,2H4A1,1,0,0,0,5,12Zm7-7a1,1,0,0,0,1-1V3a1,1,0,0,0-2,0V4A1,1,0,0,0,12,5ZM5.64,7.05a1,1,0,0,0,.7.29,1,1,0,0,0,.71-.29,1,1,0,0,0,0-1.41l-.71-.71A1,1,0,0,0,4.93,6.34Zm12,.29a1,1,0,0,0,.7-.29l.71-.71a1,1,0,1,0-1.41-1.41L17,5.64a1,1,0,0,0,0,1.41A1,1,0,0,0,17.66,7.34ZM21,11H20a1,1,0,0,0,0,2h1a1,1,0,0,0,0-2Zm-9,8a1,1,0,0,0-1,1v1a1,1,0,0,0,2,0V20A1,1,0,0,0,12,19ZM18.36,17A1,1,0,0,0,17,18.36l.71.71a1,1,0,0,0,1.41,0,1,1,0,0,0,0-1.41ZM12,6.5A5.5,5.5,0,1,0,17.5,12,5.51,5.51,0,0,0,12,6.5Zm0,9A3.5,3.5,0,1,1,15.5,12,3.5,3.5,0,0,1,12,15.5Z"
          />
        </svg>

        <!-- moon icon -->
        <svg
          class="swap-on h-10 w-10 fill-current"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
        >
          <path
            d="M21.64,13a1,1,0,0,0-1.05-.14,8.05,8.05,0,0,1-3.37.73A8.15,8.15,0,0,1,9.08,5.49a8.59,8.59,0,0,1,.25-2A1,1,0,0,0,8,2.36,10.14,10.14,0,1,0,22,14.05,1,1,0,0,0,21.64,13Zm-9.5,6.69A8.14,8.14,0,0,1,7.08,5.22v.27A10.15,10.15,0,0,0,17.22,15.63a9.79,9.79,0,0,0,2.1-.22A8.11,8.11,0,0,1,12.14,19.73Z"
          />
        </svg>
      </label>
    </div>
  </div>
</div>

<!-- Menu in the center -->
<div class="fixed left-1/2 top-0 -translate-x-1/2 flex justify-center pt-2">
  <ul
    class="menu menu-horizontal gap-2 dark:bg-base-200 bg-zinc-400 rounded-box"
  >
    {#each menuBarStates as option}
      <li>
        <button
          onclick={() =>
            (currentPage.value = {
              name: option.name,
            })}
          class={currentPage.value?.name === option.name ? "active" : ""}
        >
          {#if option.icon && option.name}
            <div>
              <Icon icon={option.icon} class="w-6 h-6" />
            </div>
          {:else}
            {option.name}
          {/if}
        </button>
      </li>
    {/each}
  </ul>
</div>
