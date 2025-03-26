<script lang='ts'>
import Algorithms from '$lib/components/Algorithms.svelte';
import NoteSelection from '$lib/components/NoteSelection.svelte';
import Sliders from '$lib/components/Sliders.svelte';


</script>

<svelte:head>
	<link rel="stylesheet" href="https://unpkg.com/aos@next/dist/aos.css" />
</svelte:head>

<div class="h-screen w-screen flex flex-col items-center justify-center bg-[#121212]">
  <div class="h-1/2 w-1/2 flex items-center justify-center space-x-12">
    <Algorithms/>
    <Sliders/>
  </div>
  <div class="h-1/2 w-full flex items-center justify-center">
    <NoteSelection/>
  </div>
</div>
