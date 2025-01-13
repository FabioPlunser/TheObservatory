import Cameras from "$lib/components/Cameras.svelte";
import Alarms from "$lib/components/Alarms.svelte";
import Rooms from "$lib/components/Rooms.svelte";
import KnownFaces from "$lib/components/KnownFaces.svelte";
import Settings from "$lib/components/Settings.svelte";

export function getPages(): any {
  let pages = $state([
    {
      name: "Cameras",
      component: Cameras,
      icon: undefined,
    },
    {
      name: "Alarms",
      component: Alarms,
      icon: undefined,
    },
    {
      name: "Known Faces",
      component: KnownFaces,
      icon: undefined,
    },
    {
      name: "Settings",
      component: Settings,
      icon: "fluent:settings-16-filled",
    },
  ]);
  return {
    get value() {
      return pages;
    }
  }
}