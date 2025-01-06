<script lang="ts">
  import { onMount } from "svelte";
  import Icon from "@iconify/svelte";

  const SERVER_URL = import.meta.env.VITE_SERVER_URL;

  async function uploadImages(event) {
    event.preventDefault();
    const formData = new FormData();
    const files = event.target.querySelector('input[type="file"]').files;
    if (!files.length) {
      alert("No files selected");
      return;
    }
    // Multiple files need to be appended with the same key name
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]); // Changed from "images" to "files" to match FastAPI
    }

    try {
      const res = await fetch(`${SERVER_URL}/api/faces/known/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        alert("Upload failed");
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      console.log(data);

      await getKnwonFaces();
    } catch (error) {
      console.error("Upload failed:", error);
      throw error;
    }
  }

  let knownFaces = $state([]);
  async function getKnwonFaces() {
    const res = await fetch(`${SERVER_URL}/api/faces/known/all`);
    const data = await res.json();
    console.log(data);
    knownFaces = data.faces;
  }

  async function deleteFace(key: any) {
    const res = await fetch(`${SERVER_URL}/api/faces/known/delete?key=${key}`, {
      method: "POST",
    });
    const data = await res.json();
    console.log(data);
    await getKnwonFaces();
  }

  onMount(async () => {
    await getKnwonFaces();
  });
</script>

<div class="mx-4">
  <div class="flex justify-center mx-auto">
    <form onsubmit={uploadImages}>
      <div class="flex items-center justify-center gap-2">
        <input
          class="file-input w-full max-w-xs"
          type="file"
          id="Image"
          name="Image"
          accept="image/*"
          multiple
          placeholder="Upload Image"
        />
        <button type="submit" class="btn btn-primary"
          >Upload selected images</button
        >
      </div>
    </form>
  </div>

  <div class="pt-24"></div>
  {#if knownFaces && knownFaces.length > 0}
    <div class="grid grid-cols-3">
      {#each knownFaces as face}
        <div class="relative">
          <div class="absolute top-0 right-0 p-4">
            <button
              onclick={async () => {
                await deleteFace(face?.key);
              }}
              ><Icon
                class="text-red-600 shadow-lg"
                icon="material-symbols:delete"
                width="24"
                height="24"
              /></button
            >
          </div>
          <img
            src={face?.url}
            alt="placeholder"
            class="rounded-lg w-full h-full"
          />
        </div>
      {/each}
    </div>
  {:else}
    <h1 class="flex justify-center mx-auto font-bold text-2xl">
      No known faces found
    </h1>
  {/if}
</div>
