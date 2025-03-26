<script lang="ts">
  import { Slider } from '@skeletonlabs/skeleton-svelte';

  let scale = $state([40]);
  let note = $state([40]);
  let octave = $state([40]);
</script>

<div class="h-full w-full flex flex-col items-center justify-center">
  <section class="w-full space-y-8">
    <p>Scale</p>
    <Slider {scale}
      onValueChange={(e) => (value = e.value)}
      meterBg="bg-primary-500" 
      thumbRingColor="ring-primary-500" 
    />
    <p>Note</p>
    <Slider {note} 
      onValueChange={(e) => (value = e.value)}
      meterBg="bg-secondary-500" 
      thumbRingColor="ring-secondary-500" 
    />
    <p>Octave</p>
    <Slider {octave}
      onValueChange={(e) => (value = e.value)}
      meterBg="bg-tertiary-500" 
      thumbRingColor="ring-tertiary-500" 
    />
  </section>
</div>
