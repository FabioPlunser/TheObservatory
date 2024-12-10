// Define a more specific type for our storage
let globalStorage = $state<Record<string, any>>({});

export function getState<T>(key: string, initialValue?: T) {
  // Initialize only if the key doesn't exist and initialValue is provided
  if (!(key in globalStorage) && initialValue !== undefined) {
    globalStorage[key] = initialValue;
  }

  return {
    get value(): T | undefined {
      return globalStorage[key];
    },
    set value(newValue: T) {
      globalStorage[key] = newValue;
    },
    delete() {
      delete globalStorage[key];
    }
  };
}
