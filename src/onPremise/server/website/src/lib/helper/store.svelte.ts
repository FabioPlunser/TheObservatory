import { browser } from "$app/environment";

export function persistentStore(key: string, value: any) {
  let storage = $state({ value });

  if (browser) {
    const item = localStorage.getItem(key);
    if (item) storage.value = JSON.parse(item);
  }

  $effect(() => {
    localStorage.setItem(key, JSON.stringify(storage.value));
  });

  return storage;
}