import { browser } from '$app/environment';

let globalStorage = $state<Record<string, any>>({});

export function persistentState<T>(key: string, initialValue?: T) {
  if (!(key in globalStorage) && initialValue !== undefined) {
    globalStorage[key] = initialValue;

    if (browser) {
      const storedValue = localStorage.getItem(key);
      if (storedValue) {
        try {
          globalStorage[key] = JSON.parse(storedValue);
        } catch (e) {
          console.warn(`Failed to parse stored value for key "${key}"`, e);
        }
      } else {
        localStorage.setItem(key, JSON.stringify(initialValue));
      }
    }
  }

  $effect(() => {
    if (browser && key in globalStorage) {
      localStorage.setItem(key, JSON.stringify(globalStorage[key]));
    }
  });

  return {
    get value(): T | undefined {
      return globalStorage[key];
    },
    set value(newValue: T) {
      globalStorage[key] = newValue;
    },
    delete() {
      delete globalStorage[key];
      if (browser) {
        localStorage.removeItem(key);
      }
    }
  };
}
