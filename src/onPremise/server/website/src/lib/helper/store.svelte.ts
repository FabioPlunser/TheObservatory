import { browser } from "$app/environment";

let storage = $state([]);

export function persistentStore(key: string, value: any) {
  let storage = $state({ value });

  if (browser) {
    const item = localStorage.getItem(key);
    if (item) storage.value = JSON.parse(item);
    if (!storage.value) {
      document.documentElement.classList.toggle('dark', true);
    }
  }

  $effect(() => {
    localStorage.setItem(key, JSON.stringify(storage.value));
  });

  return storage;
}